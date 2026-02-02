
module top_lrf_odins #(
    // ============================================================
    // Feature Map / 输入空间尺寸
    // ============================================================
    parameter FM_W   = 16,   // 输入特征图宽度（x 方向）
    parameter FM_H   = 16,   // 输入特征图高度（y 方向）
    parameter FM_C   = 3,    // 输入特征图通道数
    // ============================================================
    // Core 阵列尺寸（通常 = 输出 feature map）
    // ============================================================
    parameter CORE_W = 8,   // 核心阵列宽度
    parameter CORE_H = 8,   // 核心阵列高度
    parameter CORE_C  = 8,   // 核心阵列通道数
    // ============================================================
    // Local Receptive Field（局部感受野）
    // ============================================================
    parameter LRF_W  = 3,    // LRF 在 x 方向大小（如 3×3）
    parameter LRF_H  = 3,     // LRF 在 y 方向大小
    parameter TIME_STEP = 8,
    parameter POST_NEUR_MEM_WIDTH = 13, // 单个突触后神经元膜电位数据位宽
    parameter WEIGHT_WIDTH = 9, // 单个突触权重数据位宽
    parameter GRAD_WIDTH = 9,
    parameter GOODNESS_WIDTH = 20
)(
    input  wire clk,        // 全局时钟
    input  wire rst,        // 全局复位（高有效）

    // ============================================================
    // 输入 AER（事件地址表示输入空间位置）
    // ============================================================
    input  wire                          AERIN_REQ,   // 输入事件请求
    input  wire [2 + $clog2(FM_C) + $clog2(FM_H) + $clog2(FM_W)-1:0]  AERIN_ADDR,  // 输入事件地址（x,y 编码）
    output wire                          AERIN_ACK,   // 输入事件应答
    // ============================================================
    // 输出 AER（事件地址表示输出空间位置）
    // ============================================================
    output wire                          AEROUT_REQ,  // 输出事件请求
    output wire [2 + $clog2(CORE_C) + $clog2(CORE_W*CORE_H) - 1:0] AEROUT_ADDR, // 输出事件地址（x,y 编码）
    input  wire                          AEROUT_ACK,  // 输出事件应答
    // ============================================================
    // 全局控制信号（广播到所有核心）
    // ============================================================
    input  wire                          IS_POS,       // 正 / 负 STDP
    input  wire                          IS_TRAIN,     // 0: 推理 1: 训练
    output wire                          ONE_SAMPLE_FINISH
);

    localparam POST_NEUR_PARALLEL = (CORE_C > 8) ? 8 : 4; //并行度
    localparam CORE_NUM = CORE_W * CORE_H;// Core 总数量
    // 每个 core 对应一个空间位置（输出 feature map）
    localparam MAP_IN_AER_WIDTH = 2 + $clog2(FM_C) + $clog2(FM_H) + $clog2(FM_W); // 映射前 AER 宽度
    // localparam MAP_OUT_AER_WIDTH = $clog2(LRF_W) + $clog2(LRF_W) + $clog2(FM_C) + 2;
    localparam MAP_OUT_AER_WIDTH = 2 + $clog2(FM_C) + $clog2(LRF_H) + $clog2(LRF_W); // 映射后 AER 宽度

    // 用于表示 (x,y) 展平后的地址
    localparam INPUT_NEURON = LRF_H * LRF_W * FM_C; // 每个 core 输入神经元数量
    localparam OUTPUT_NEURON = CORE_C; // 每个 core 输出神经元数量
    localparam AER_IN_WIDTH = MAP_OUT_AER_WIDTH; // 每个 core 输入AER 地址宽度
    localparam AER_OUT_WIDTH = 2 + $clog2(OUTPUT_NEURON); // 每个 core 输出 AER 输出地址宽度
    // 该层的输出 AER 宽度
    localparam AER_OUT_NEXT_LAYER_WIDTH = 2 + $clog2(CORE_C) + $clog2(CORE_W*CORE_H);
    // 突触前神经元地址宽度
    localparam PRE_NEUR_ADDR_WIDTH = MAP_OUT_AER_WIDTH - 2; // 突触前神经元地址宽度
    localparam PRE_NEUR_WORD_ADDR_WIDTH= MAP_OUT_AER_WIDTH - 2;
    localparam PRE_NEUR_BYTE_ADDR_WIDTH = 0;
    localparam POST_NEUR_ADDR_WIDTH = $clog2(CORE_C);
    localparam POST_NEUR_BYTE_ADDR_WIDTH = $clog2(POST_NEUR_PARALLEL);
    localparam POST_NEUR_WORD_ADDR_WIDTH= POST_NEUR_ADDR_WIDTH - POST_NEUR_BYTE_ADDR_WIDTH;
    localparam SYN_ARRAY_DATA_WIDTH = POST_NEUR_PARALLEL * WEIGHT_WIDTH; // 突触阵列数据位宽
    localparam SYN_ARRAY_ADDR_WIDTH = $clog2(INPUT_NEURON) + $clog2(OUTPUT_NEURON) - $clog2(POST_NEUR_PARALLEL); // 突触阵列地址位宽 
    localparam GRAD_ARRAY_DATA_WIDTH = POST_NEUR_PARALLEL * GRAD_WIDTH; // 突触梯度阵列数据位宽
    localparam GRAD_ARRAY_ADDR_WIDTH = $clog2(INPUT_NEURON) + $clog2(OUTPUT_NEURON) - $clog2(POST_NEUR_PARALLEL); // 突触梯度阵列地址位宽
    localparam POST_NEUR_SPIKE_CNT_WIDTH = TIME_STEP; // 单个突触后神经元脉冲计数数据位宽
    localparam PRE_NEUR_DATA_WIDTH = TIME_STEP; // 单个突触前神经元脉冲计数数据位宽
    localparam POST_NEUR_DATA_WIDTH = 1 + POST_NEUR_SPIKE_CNT_WIDTH + POST_NEUR_MEM_WIDTH; // 单个突触后神经元状态数据位宽


    // ============================================================
    // Mapper 输出给 Core 的 AER 信号
    // ============================================================
    // 每个 core 是否被当前事件命中（LRF 内）
    wire [CORE_NUM-1:0] core_aer_req;
    wire [CORE_NUM-1:0] core_aer_out_req;

    // 每个 core 对应的 AER ACK
    // 所有命中 core ACK 后，mapper 才会 ACK 上游
    wire [CORE_NUM-1:0] core_aer_ack;
    wire [CORE_NUM-1:0] core_aer_out_ack;

    // 每个 core 接收到的 AER 地址
    // 通常是同一个 AERIN_ADDR（广播）
    wire [CORE_NUM-1:0][MAP_OUT_AER_WIDTH-1:0] core_aer_event;
    wire [CORE_NUM*AER_OUT_WIDTH-1:0] core_aer_out_event;



    // ============================================================
    // AER Local Receptive Field Mapper
    // 功能：
    //   1. 根据 AERIN_ADDR 计算 LRF 覆盖范围
    //   2. 对命中的 core 产生并行 AER_REQ
    //   3. 汇聚所有 core 的 ACK，保证事件一致性
    // ============================================================
    wire                        MAP_IN_AERIN_REQ = AERIN_REQ;
    wire [MAP_IN_AER_WIDTH-1:0] MAP_IN_AERIN_EVENT = AERIN_ADDR;
    wire [MAP_IN_AER_WIDTH-2-1:0]MAP_IN_AERIN_IDX = AERIN_ADDR[MAP_IN_AER_WIDTH-2-1:0];
    wire                     	MAP_IN_AERIN_ACK;
    assign                      AERIN_ACK = MAP_IN_AERIN_ACK;
    

    wire [CORE_W*CORE_H-1:0] 	MAP_OUT_AERIN_REQ;
    assign                      core_aer_req = MAP_OUT_AERIN_REQ;
    wire [CORE_W*CORE_H-1:0][MAP_OUT_AER_WIDTH-1: 0] MAP_OUT_AERIN_EVENT;
    assign                      core_aer_event = MAP_OUT_AERIN_EVENT;
    wire [CORE_W*CORE_H-1:0] 	MAP_OUT_AERIN_IDX;
    wire [CORE_W*CORE_H-1:0]    MAP_OUT_AERIN_ACK = core_aer_ack;

    aer_in_lrf_mapper #(
        .MAP_IN_AER_WIDTH (MAP_IN_AER_WIDTH),
        .MAP_OUT_AER_WIDTH(MAP_OUT_AER_WIDTH),
        .FM_C(FM_C),
        .FM_W(FM_W),
        .FM_H(FM_H),
        .CORE_W(CORE_W),
        .CORE_H(CORE_H),
        .CORE_C(CORE_C),
        .LRF_W(LRF_W),
        .LRF_H(LRF_H)
    ) u_aer_lrf_mapper (
        .clk(clk),
        .rst(rst),
        // 上游 AER 接口
        .MAP_IN_AERIN_REQ(MAP_IN_AERIN_REQ),
        .MAP_IN_AERIN_EVENT(MAP_IN_AERIN_EVENT),
        .MAP_IN_AERIN_IDX(MAP_IN_AERIN_IDX),
        .MAP_IN_AERIN_ACK(MAP_IN_AERIN_ACK),
        // 下游（core）AER 接口
        .MAP_OUT_AERIN_REQ(MAP_OUT_AERIN_REQ),
        .MAP_OUT_AERIN_EVENT(MAP_OUT_AERIN_EVENT),
        .MAP_OUT_AERIN_IDX(MAP_OUT_AERIN_IDX),
        .MAP_OUT_AERIN_ACK(MAP_OUT_AERIN_ACK)
    );

    // ============================================================
    // ODIN Core Array
    // 每个 core：
    //   - 实现一个 M×N 全连接 / 局部卷积单元
    //   - 独立 STDP / 神经元状态
    // ============================================================
    wire [CORE_NUM-1:0]                core_valid;
    wire [CORE_NUM-1:0]                core_clear_goodness;
    wire [CORE_NUM * POST_NEUR_PARALLEL * POST_NEUR_MEM_WIDTH-1:0] core_mem_bus;
    wire [CORE_NUM * GOODNESS_WIDTH - 1 :0] avg_mem_bus;
    genvar i;
    generate
        for (i = 0; i < CORE_NUM; i = i + 1) begin : ODIN_ARRAY

            // ----------------------------------------------------
            // core 内部输出状态（可用于监控 / 聚合）
            // ----------------------------------------------------
            wire [GOODNESS_WIDTH-1:0] GOODNESS;            // 本 core 的 goodness 累积
            // wire        ONE_SAMPLE_FINISH;   // 单样本处理完成
            wire        GOODNESS_ACC_VALID;  // goodness 更新有效

            // ----------------------------------------------------
            // ODIN FF-STDP Core
            // ----------------------------------------------------
            // outports wire
            wire                                              	AERIN_ACK;
            wire [MAP_OUT_AER_WIDTH-1:0]                        AEROUT_ADDR;
            wire                                              	AEROUT_REQ;
            wire [POST_NEUR_MEM_WIDTH*POST_NEUR_PARALLEL-1:0] 	POST_NEUR_MEM_BUS;
            wire                                              	GOODNESS_CLEAR;
            wire        [GOODNESS_WIDTH-1:0] AVG_GOODNESS;
            ODIN_ffstdp
            #(
                .TIME_STEP                             (TIME_STEP          ),
                .INPUT_NEURON                          (INPUT_NEURON       ),
                .OUTPUT_NEURON                         (OUTPUT_NEURON      ),
                .AER_IN_WIDTH                          (AER_IN_WIDTH          ),
                .AER_OUT_WIDTH                         (AER_OUT_WIDTH          ),
                .PRE_NEUR_ADDR_WIDTH                   (PRE_NEUR_ADDR_WIDTH),
                .PRE_NEUR_WORD_ADDR_WIDTH              (PRE_NEUR_WORD_ADDR_WIDTH),
                .PRE_NEUR_BYTE_ADDR_WIDTH              (PRE_NEUR_BYTE_ADDR_WIDTH),
                .POST_NEUR_ADDR_WIDTH                  (POST_NEUR_ADDR_WIDTH),
                .POST_NEUR_WORD_ADDR_WIDTH             (POST_NEUR_WORD_ADDR_WIDTH),
                .POST_NEUR_BYTE_ADDR_WIDTH             (POST_NEUR_BYTE_ADDR_WIDTH),
                .POST_NEUR_PARALLEL                    (POST_NEUR_PARALLEL ),
                .PRE_NEUR_DATA_WIDTH                   (PRE_NEUR_DATA_WIDTH),
                .POST_NEUR_DATA_WIDTH                  (POST_NEUR_DATA_WIDTH),
                .POST_NEUR_MEM_WIDTH                   (POST_NEUR_MEM_WIDTH),
                .POST_NEUR_SPIKE_CNT_WIDTH             (POST_NEUR_SPIKE_CNT_WIDTH),
                .SYN_ARRAY_DATA_WIDTH                  (SYN_ARRAY_DATA_WIDTH),
                .SYN_ARRAY_ADDR_WIDTH                  (SYN_ARRAY_ADDR_WIDTH),
                .GRAD_ARRAY_DATA_WIDTH                 (GRAD_ARRAY_DATA_WIDTH),
                .GRAD_ARRAY_ADDR_WIDTH                 (GRAD_ARRAY_ADDR_WIDTH),
                .WEIGHT_WIDTH                          (WEIGHT_WIDTH),
                .GRAD_WIDTH                            (GRAD_WIDTH),
                .GOODNESS_WIDTH                        (GOODNESS_WIDTH)
            )
            u_core (
                // ---------- Global ----------
                .CLK                    (clk),
                .RST                    (rst),
                .IS_POS                 (IS_POS),
                .IS_TRAIN               (IS_TRAIN),
                .AVG_GOODNESS           (AVG_GOODNESS),

                // ---------- AER Input ----------
                .AERIN_ADDR             (core_aer_event[i]), // 来自 mapper
                .AERIN_REQ              (core_aer_req[i]),  // 是否命中该 core
                .AERIN_ACK              (core_aer_ack[i]),  // core 接收完成

                // ---------- AER Output（可级联） ----------
                .AEROUT_ADDR            (AEROUT_ADDR),
                .AEROUT_REQ             (AEROUT_REQ),
                .AEROUT_ACK             (AERIN_ACK),

                // ---------- Status ----------
                .GOODNESS               (GOODNESS),
                .ONE_SAMPLE_FINISH      (ONE_SAMPLE_FINISH),
                .POST_NEUR_MEM_BUS      (POST_NEUR_MEM_BUS),
                .GOODNESS_ACC_VALID     (GOODNESS_ACC_VALID),
                .GOODNESS_CLEAR         (GOODNESS_CLEAR)
            );
            // AER Output
            assign core_aer_out_event[i*AER_OUT_WIDTH +: AER_OUT_WIDTH] = AEROUT_ADDR;
            assign core_aer_out_req[i] = AEROUT_REQ;
            assign AERIN_ACK = core_aer_out_ack[i];

            // GOODNESS 移动平均模块连接
            assign core_valid[i] = GOODNESS_ACC_VALID;
            assign core_clear_goodness[i] = GOODNESS_CLEAR;
            assign core_mem_bus[i*POST_NEUR_PARALLEL*POST_NEUR_MEM_WIDTH +: POST_NEUR_PARALLEL*POST_NEUR_MEM_WIDTH] = POST_NEUR_MEM_BUS;
            assign AVG_GOODNESS = avg_mem_bus[i*GOODNESS_WIDTH +: GOODNESS_WIDTH];
        end
    endgenerate

    goodness_moving_avg #(
	.CORE_NUM            	( CORE_NUM   ),
	.POST_NEUR_PARALLEL  	( POST_NEUR_PARALLEL    ),
	.POST_NEUR_MEM_WIDTH 	( POST_NEUR_MEM_WIDTH   ),
	.GOODNESS_WIDTH      	( GOODNESS_WIDTH   ),
	.AVG_SHIFT           	( OUTPUT_NEURON / POST_NEUR_PARALLEL  )
    )
    u_goodness_moving_avg(
        .clk                 	( clk                  ),
        .rst                 	( rst                  ),
        .core_valid          	( core_valid          ),
        .core_clear_goodness 	( core_clear_goodness  ),
        .core_mem_bus        	( core_mem_bus    ),
        .avg_mem_bus         	( avg_mem_bus         )
    );
    // outports wire
    wire                                	    evt_req;
    wire [AER_OUT_NEXT_LAYER_WIDTH-1:0] 	    evt_addr;
    wire                                	    evt_ack;

    aer_core_event_arbiter #(
        .CORE_NUM      	(CORE_NUM),
        .AER_OUT_NEXT_LAYER_WIDTH (AER_OUT_NEXT_LAYER_WIDTH),
        .AER_OUT_CORE_WIDTH 	(AER_OUT_WIDTH))
    u_aer_core_event_arbiter(
        .clk       	( clk        ),
        .rst       	( rst        ),
        .core_req  	( core_aer_out_req   ),
        .core_addr 	( core_aer_out_event  ),
        .core_ack  	( core_aer_out_ack   ),
        .evt_req   	( evt_req    ),
        .evt_addr  	( evt_addr   ),
        .evt_ack   	( evt_ack    )
    );

    assign AEROUT_ADDR = evt_addr;
    assign AEROUT_REQ = evt_req;
    assign evt_ack = AEROUT_ACK;

endmodule

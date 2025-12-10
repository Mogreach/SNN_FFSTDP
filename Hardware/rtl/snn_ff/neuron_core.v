module neuron_core #(
    parameter TIME_STEP = 8,
    parameter INPUT_NEURON = 784,
    parameter OUTPUT_NEURON = 256,
    parameter AER_WIDTH = 12,

    parameter PRE_NEUR_ADDR_WIDTH = 10,
    parameter PRE_NEUR_WORD_ADDR_WIDTH= 10,
    parameter PRE_NEUR_BYTE_ADDR_WIDTH = 0,

    parameter POST_NEUR_ADDR_WIDTH = 10,
    parameter POST_NEUR_WORD_ADDR_WIDTH= 8,
    parameter POST_NEUR_BYTE_ADDR_WIDTH = 2,
    parameter POST_NEUR_PARALLEL = 4,
    
    parameter PRE_NEUR_DATA_WIDTH = 8,
    parameter POST_NEUR_DATA_WIDTH = 32,
    parameter POST_NEUR_MEM_WIDTH = 12,
    parameter POST_NEUR_SPIKE_CNT_WIDTH = 7,
    parameter SYN_ARRAY_DATA_WIDTH = 32,
    parameter SYN_ARRAY_ADDR_WIDTH = 16,
    parameter WEIGHT_WIDTH = 8
)(
    // Global inputs ------------------------------------------
    input  wire CLK,
    input  wire RST_N,
    // Synaptic inputs ----------------------------------------
    input  wire [SYN_ARRAY_DATA_WIDTH-1:0] SYNARRAY_RDATA,
    // Controller inputs ----------------------------------------
        // SPI控制编入数据
    input wire  [POST_NEUR_DATA_WIDTH-1:0] CTRL_POST_NEUR_PROG_DATA,
        //控制器突触地址
    input wire [PRE_NEUR_ADDR_WIDTH-1:0] CTRL_PRE_NEURON_ADDRESS,
    input wire [POST_NEUR_ADDR_WIDTH-1:0] CTRL_POST_NEURON_ADDRESS,
        //SPI控制读写事件
    input wire       CTRL_WR_NEUR_EVENT,
    input wire       CTRL_RD_NEUR_EVENT,
        //训练推理事件
    input wire  CTRL_NEUR_EVENT,
    input wire  CTRL_TSTEP_EVENT,
    input wire  CTRL_TREF_EVENT,

    input wire  CTRL_PRE_NEUR_CS,
    input wire  CTRL_PRE_NEUR_WE,
    input wire  CTRL_POST_NEUR_CS,
    input wire  CTRL_POST_NEUR_WE,
    input wire  CTRL_PRE_CNT_EN,
    // SPI inputs
    input wire  SPI_GATE_ACTIVITY_sync,
    input wire  [POST_NEUR_ADDR_WIDTH-1:0] SPI_POST_NEUR_ADDR,

    // Outputs
    output reg [POST_NEUR_DATA_WIDTH-1:0] NEUR_STATE,
    output wire [POST_NEUR_PARALLEL-1:0] NEUR_EVENT_OUT,
    output wire [PRE_NEUR_DATA_WIDTH-1:0] PRE_NEUR_S_CNT,
    output wire [POST_NEUR_SPIKE_CNT_WIDTH * POST_NEUR_PARALLEL -1 :0] POST_NEUR_S_CNT
);

    localparam POST_NEUR_SRAM_DATA_WIDTH = POST_NEUR_PARALLEL * POST_NEUR_DATA_WIDTH; 
    
    // localparam  neuron_thresold= 12'b0_00001_100000; //神经元阈值S5.6: 1.5
    localparam  neuron_thresold= 12'b0000_0100_1101; // 神经元阈值S5.6: 1.2
    // localparam  neuron_thresold= 12'b0_00001_000000; //神经元阈值S5.6: 1
    // Internal regs and wires definitions
    wire [PRE_NEUR_DATA_WIDTH-1:0] pre_neuron_sram_out;
    wire [PRE_NEUR_DATA_WIDTH-1:0] pre_neuron_sram_in;

    wire [POST_NEUR_SRAM_DATA_WIDTH-1:0] post_neuron_sram_out;
    wire [POST_NEUR_SRAM_DATA_WIDTH-1:0] post_neuron_sram_in;
    wire [POST_NEUR_DATA_WIDTH-1:0] post_neuron_sram_in_array [0:POST_NEUR_PARALLEL-1];
    wire [POST_NEUR_DATA_WIDTH-1:0] post_neuron_sram_out_array [0:POST_NEUR_PARALLEL-1];

    wire [POST_NEUR_WORD_ADDR_WIDTH - POST_NEUR_BYTE_ADDR_WIDTH-1:0]  post_neuron_sram_addr;
    wire [POST_NEUR_BYTE_ADDR_WIDTH-1:0]  post_neuron_byte_addr;
    wire [POST_NEUR_PARALLEL-1:0]  IF_neuron_event_out;


    // assign NEUR_STATE = (post_neuron_sram_out >> ({5'b0,post_neuron_byte_addr} << 5))[31:0]; //右移post_neuron_byte_address * 32
    assign post_neuron_sram_addr = CTRL_POST_NEURON_ADDRESS[POST_NEUR_WORD_ADDR_WIDTH-1 : POST_NEUR_BYTE_ADDR_WIDTH];
    assign post_neuron_byte_addr = CTRL_POST_NEURON_ADDRESS[POST_NEUR_BYTE_ADDR_WIDTH-1:0];
    assign PRE_NEUR_S_CNT = pre_neuron_sram_out;




    always @(*) begin
        NEUR_STATE = post_neuron_sram_out_array[post_neuron_byte_addr];
    end



    // reg neur_event_d;
    // always @(posedge CLK or negedge RST_N)           
    //     begin                                        
    //         if(!RST_N)                               
    //             neur_event_d <= 0;                             
    //         else                                    
    //             neur_event_d <= CTRL_NEUR_EVENT; 
    //     end                                          

    pre_neuron pre_neuron_0( 
    .pre_spike_cnt(pre_neuron_sram_out),          // 突触前神经元发放脉冲数量 from SRAM
    .neuron_event(CTRL_NEUR_EVENT),               // synaptic event trigger
    .neuron_event_pulse(CTRL_PRE_CNT_EN),
    .time_ref_event(CTRL_TREF_EVENT),                // time reference event trigger
    .pre_spike_cnt_next(pre_neuron_sram_in)          // 突触前神经元发放脉冲数量 to SRAM
);

    genvar i;
    // 神经元状态更新模块 + SPI初始化
    generate
        for (i=0; i < POST_NEUR_PARALLEL; i=i+1) begin : gen_post_neuron
            // 神经元状态信息更新：SPI 配置？（SPI指定地址？掩码后的编入数据：保持）：膜电位更新
            // 突触后神经元膜电位更新
            // assign post_neuron_sram_in[11+(i*32): 0+(i*32)] = SPI_GATE_ACTIVITY_sync ? CTRL_POST_NEUR_PROG_DATA[11:0]: IF_neuron_next_mem[i]; 
            wire [POST_NEUR_MEM_WIDTH-1:0]  post_neuron_mem_next;
            // 突触后神经元阈值更新
            // assign post_neuron_sram_in[23+(i*32):12+(i*32)] = SPI_GATE_ACTIVITY_sync ? CTRL_POST_NEUR_PROG_DATA[23:12] : neuron_thresold;
            wire [POST_NEUR_MEM_WIDTH-1:0]  post_neuron_thresold_next = neuron_thresold;
            // 突触后神经元发放脉冲更新
            // assign post_neuron_sram_in[30+(i*32):24+(i*32)] = SPI_GATE_ACTIVITY_sync ? CTRL_POST_NEUR_PROG_DATA[30:24] : IF_neuron_next_spike_cnt[i];
            wire [POST_NEUR_SPIKE_CNT_WIDTH-1:0]  post_neuron_spike_cnt_next;
            // 突触后神经元使能信号更新                 
            // assign post_neuron_sram_in[31+(i*32)] = SPI_GATE_ACTIVITY_sync ? CTRL_POST_NEUR_PROG_DATA[31] : post_neuron_sram_out[31+(i*32)];                            
            // assign post_neuron_sram_in[31+(i*32)] = SPI_GATE_ACTIVITY_sync ? CTRL_POST_NEUR_PROG_DATA[31] : 1'b1;
            wire post_neuron_enable_next = 1'b1;
            assign post_neuron_sram_in_array[i] = {post_neuron_enable_next, post_neuron_spike_cnt_next, post_neuron_thresold_next, post_neuron_mem_next};
            assign post_neuron_sram_in[i*POST_NEUR_DATA_WIDTH +: POST_NEUR_DATA_WIDTH] = post_neuron_sram_in_array[i];

            assign post_neuron_sram_out_array[i] = post_neuron_sram_out[i*POST_NEUR_DATA_WIDTH +: POST_NEUR_DATA_WIDTH];
            wire [POST_NEUR_MEM_WIDTH-1:0]  post_neuron_mem = post_neuron_sram_out_array[i][POST_NEUR_MEM_WIDTH-1:0];
            wire [POST_NEUR_MEM_WIDTH-1:0]  post_neuron_thresold = post_neuron_sram_out_array[i][POST_NEUR_MEM_WIDTH +: POST_NEUR_MEM_WIDTH];
            wire [POST_NEUR_SPIKE_CNT_WIDTH-1:0]  post_neuron_spike_cnt = post_neuron_sram_out_array[i][POST_NEUR_MEM_WIDTH + POST_NEUR_MEM_WIDTH +: POST_NEUR_SPIKE_CNT_WIDTH];
            wire post_neuron_enable = post_neuron_sram_out_array[i][POST_NEUR_DATA_WIDTH-1];

            if_neuron #(
                .AER_WIDTH                             (AER_WIDTH          ),
                .POST_NEUR_MEM_WIDTH                   (POST_NEUR_MEM_WIDTH),
                .POST_NEUR_SPIKE_CNT_WIDTH             (POST_NEUR_SPIKE_CNT_WIDTH),
                .WEIGHT_WIDTH                          (WEIGHT_WIDTH       ) 
            )
            if_neuron_gen(
                .CLK                                   (CLK                ),
                .post_spike_cnt                        (post_neuron_spike_cnt),// 突触后神经元发放脉冲数量 from SRAM
                .post_spike_cnt_next                   (post_neuron_spike_cnt_next),// 突触后神经元发放脉冲数量 to SRAM
                .param_thr                             (post_neuron_thresold),// neuron firing threshold parameter 
                .state_core                            (post_neuron_mem    ),// core neuron state from SRAM 
                .state_core_next                       (post_neuron_mem_next),// next core neuron state to SRAM
                .syn_weight                            (SYNARRAY_RDATA [i*WEIGHT_WIDTH +: WEIGHT_WIDTH]),// synaptic weight
                .neuron_event                          (CTRL_NEUR_EVENT    ),// synaptic event trigger
                .time_step_event                       (CTRL_TSTEP_EVENT   ),
                .time_ref_event                        (CTRL_TREF_EVENT    ),// time reference event trigger
                .spike_out                             (IF_neuron_event_out[i]) // neuron spike event output  
            );
            // assign NEUR_EVENT_OUT[i] = post_neuron_enable? ((CTRL_POST_NEUR_CS && CTRL_POST_NEUR_WE) ? IF_neuron_event_out[i] : 1'b0) : 1'b0;
            assign NEUR_EVENT_OUT[i] = post_neuron_enable? IF_neuron_event_out[i] : 1'b0;
            assign POST_NEUR_S_CNT[i*POST_NEUR_SPIKE_CNT_WIDTH +: POST_NEUR_SPIKE_CNT_WIDTH] = post_neuron_spike_cnt;
        end
    endgenerate
    // 突触前神经元SRAM 读优先时序
    SRAM_1024x8_wrapper neurarray_pre (       
        
        // Global inputs
        .clk         (CLK),
    
        // Control and data inputs
        // .ena        (CTRL_PRE_NEUR_CS),
        .we         (CTRL_PRE_NEUR_WE),
        .a          (CTRL_PRE_NEURON_ADDRESS),
        .d          (pre_neuron_sram_in),
        // Data output
        .spo          (pre_neuron_sram_out)
    );
    
    // sram_wrapper#(
    // .ADDR_WIDTH     (PRE_NEUR_ADDR_WIDTH              ),
    // .DATA_WIDTH     (PRE_NEUR_DATA_WIDTH             ),
    // .SRAM_DEPTH     (INPUT_NEURON            )
    // )
    // neurarray_pre(
    // // Global inputs
    //     .CK                                 (CLK                        ),// Clock (synchronous read/write)
    // // Control and data inputs
    //     .CS                                 (CTRL_PRE_NEUR_CS                        ),// Chip select
    //     .WE                                 (CTRL_PRE_NEUR_WE                        ),// Write enable
    //     .A                                  (CTRL_PRE_NEURON_ADDRESS                         ),// Address bus
    //     .D                                  (pre_neuron_sram_in                         ),// Data input bus (write)
    // // Data output
    //     .Q                                  (pre_neuron_sram_out                        )// Data output bus (read)
    // );

    // 突触后神经元SRAM 读优先时序
    SRAM_256x128_wrapper neurarray_post (       
        
        // Global inputs
        .clka         (CLK),
    
        // Control and data inputs
        .ena         (CTRL_POST_NEUR_CS),
        .wea         (CTRL_POST_NEUR_WE),
        .addra          (post_neuron_sram_addr),
        .dina          (post_neuron_sram_in),
        // Data output
        .douta          (post_neuron_sram_out)
    );
    // sram_wrapper#(
    // .ADDR_WIDTH     (POST_NEUR_WORD_ADDR_WIDTH - POST_NEUR_BYTE_ADDR_WIDTH),
    // .DATA_WIDTH     (POST_NEUR_SRAM_DATA_WIDTH         ),
    // .SRAM_DEPTH     (OUTPUT_NEURON / POST_NEUR_PARALLEL)
    // )
    // neurarray_post(
    // // Global inputs
    //     .CK                                 (CLK                        ),// Clock (synchronous read/write)
    // // Control and data inputs
    //     .CS                                 (CTRL_POST_NEUR_CS                        ),// Chip select
    //     .WE                                 (CTRL_POST_NEUR_WE                        ),// Write enable
    //     .A                                  (post_neuron_sram_addr                         ),// Address bus
    //     .D                                  (post_neuron_sram_in                         ),// Data input bus (write)
    // // Data output
    //     .Q                                  (post_neuron_sram_out                        )// Data output bus (read)
    // );


//    Neuron_SRAM neurarray_0(
//    .clka  (CLK      ),  // input wire clka
//    .ena   (CTRL_NEURMEM_CS       ),  // input 片选使能信号
//    .wea   (CTRL_NEURMEM_WE       ),  // input 写使能信号
//    .addra (CTRL_NEURMEM_ADDR     ), 
//    .dina  (neuron_data  ),
//    .douta (NEUR_STATE  )  
//);

endmodule





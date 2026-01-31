module aer_lrf_mapper #(
    parameter MAP_IN_AER_WIDTH   = 12,   // 映射前AER 宽度
    parameter MAP_OUT_AER_WIDTH  = 12,   // 映射后AER 宽度
    parameter FM_C         = 16,   // 输入通道数
    parameter FM_W        = 32,   // 输入特征图宽
    parameter FM_H        = 32,   // 输入特征图高
    parameter CORE_W      = 16,   // 核心阵列宽
    parameter CORE_H      = 16,   // 核心阵列高
    parameter CORE_C      = 3,    // 输出通道数
    parameter LRF_W       = 3,    // 局部感受野宽
    parameter LRF_H       = 3,    // 局部感受野高
    parameter STRIDE      = 1
)(
    input  wire                     clk,
    input  wire                     rst,

    // --------- Input AER ----------
    input  wire                          MAP_IN_AERIN_REQ,
    input  wire [MAP_IN_AER_WIDTH-1:0]   MAP_IN_AERIN_EVENT,
    input  wire [MAP_IN_AER_WIDTH-2-1:0] MAP_IN_AERIN_IDX,
    output wire                     MAP_IN_AERIN_ACK,

    // --------- Output AER to cores ----------
    output wire [CORE_W*CORE_H-1:0] MAP_OUT_AERIN_REQ,
    output wire [CORE_W*CORE_H-1:0][MAP_OUT_AER_WIDTH-1: 0]    MAP_OUT_AERIN_EVENT,
    output wire [CORE_W*CORE_H-1:0][MAP_OUT_AER_WIDTH-2-1:0]   MAP_OUT_AERIN_IDX,
    input  wire [CORE_W*CORE_H-1:0] MAP_OUT_AERIN_ACK
);

    localparam X_BITS = $clog2(FM_W);
    localparam Y_BITS = $clog2(FM_H);
    localparam C_BITS = $clog2(FM_C);


    wire [X_BITS-1:0] in_x;
    wire [Y_BITS-1:0] in_y;
    wire [C_BITS-1:0] in_c;

    assign {in_c, in_y, in_x} = MAP_IN_AERIN_IDX; // 输入特征图通道、高度、宽度对应的位
    wire [1:0] event_type = MAP_IN_AERIN_EVENT[(MAP_IN_AER_WIDTH-1) -: 2];
    wire [MAP_OUT_AER_WIDTH-1: 0] map_out_aer_non_neur_event = {event_type,{MAP_OUT_AER_WIDTH{1'b1}}};
    wire [MAP_OUT_AER_WIDTH-1: 0] map_out_aer_invalid_event = {2'b11,{MAP_OUT_AER_WIDTH{1'b1}}};

    localparam RX = LRF_W / 2;
    localparam RY = LRF_H / 2;
    // 局部地址宽度
    localparam DX_W = $clog2(LRF_W);
    localparam DY_W = $clog2(LRF_H);
    
    reg [DX_W-1:0] LRF_X_idx;
    // 高位优先编码
    wire [LRF_W-1:0] LRF_x_idx_condition_priority;
    wire [LRF_H-1:0] LRF_y_idx_condition_priority;

    genvar i, j;
    generate
        // 根据in_x，in_y判断该事件会驱动多少个核心；因一个事件最多驱动LRF_W * LRF_H个核心，转换成阵列就有 LRF_W * LRF_H 个条件
        // 这些条件代表着当前事件的会出现在哪些局部感受野位置上，采用独热码编码：LRF_x_idx_condition第二位代表会出现在感受野第二行以内，LRF_y_idx_condition则代表会出现在感受野第二列以内
        // 重叠部分则代表会出现的具体位置，若为每个感受野位置都附上一个使能信号，该使能信号根据每行每列的的条件进行与逻辑
        // 这意味着需要将独热码编码成 高位优先编码
        wire [LRF_W-1:0] LRF_x_idx_condition;
        wire [LRF_H-1:0] LRF_y_idx_condition;
        // 边界条件判断：即输入像素位置落在 边界 ± LRF-1 范围内
        for (j = 0; j < LRF_W - 1; j = j + 1) begin
            assign LRF_x_idx_condition[j] = (in_x == j[X_BITS-1:0]) || (in_x == ((FM_W - 1'b1 - j)));
        end
        for (j = 0; j < LRF_H - 1; j = j + 1) begin
            assign LRF_y_idx_condition[j] = (in_y == j[Y_BITS-1:0]) || (in_y == ((FM_H - 1'b1 - j)));
        end
        // 非边界条件判断：即输入像素位置落在区域中心
        assign LRF_x_idx_condition[LRF_W-1] = !(|LRF_x_idx_condition[LRF_W-2:0]);
        assign LRF_y_idx_condition[LRF_H-1] = !(|LRF_y_idx_condition[LRF_H-2:0]);
        assign LRF_x_idx_condition_priority[LRF_W-1] = LRF_x_idx_condition[LRF_W-1];
        assign LRF_y_idx_condition_priority[LRF_H-1] = LRF_y_idx_condition[LRF_H-1];
        // 将独热条件编码，映射成 高位优先编码，
        for (j = LRF_W - 2; j >= 0; j = j - 1) begin
            assign LRF_x_idx_condition_priority[j] = LRF_x_idx_condition[j] || (|LRF_x_idx_condition_priority[LRF_W-1:j+1]);
        end
        for (j = LRF_H - 2; j >= 0; j = j - 1) begin
            assign LRF_y_idx_condition_priority[j] = LRF_y_idx_condition[j] || (|LRF_y_idx_condition_priority[LRF_H-1:j+1]);
        end
    endgenerate

    // 映射核心编号和使能
    wire [LRF_W*LRF_H-1:0] LRF_core_en;
    // 每个感受野位置下对应的核心编号[H，W]
    wire [LRF_W*LRF_H-1:0][X_BITS + Y_BITS -1:0] LRF_core_id;

    genvar ix, iy;
    generate
        for (iy = 0; iy < LRF_H; iy = iy + 1) begin : GEN_Y
            for (ix = 0; ix < LRF_W; ix = ix + 1) begin : GEN_X
                localparam int LRF_IDX = iy * LRF_W + ix;
                // 相对感受野中心的偏移
                localparam offset_x = ix - RX;
                localparam offset_y = iy - RY;
                wire [X_BITS-1:0] map_out2core_x = in_x + offset_x;
                wire [Y_BITS-1:0] map_out2core_y = in_y + offset_y;
                // [H，W]紧凑型编号
                wire [MAP_OUT_AER_WIDTH-2-1:0] core_id = map_out2core_y * CORE_W + map_out2core_x;
                // ============ 核心使能 ============
                assign LRF_core_en[LRF_IDX] = LRF_x_idx_condition_priority[ix] & LRF_y_idx_condition_priority[iy];
                // ============ 核心编号 ============
                assign LRF_core_id[LRF_IDX] = core_id;
                // assign LRF_core_id[LRF_IDX] = {map_out2core_y, map_out2core_x};
            end
        end
    endgenerate

    reg [CORE_H*CORE_W-1:0] MAP_OUT_AERIN_REQ_r;
    reg [MAP_OUT_AER_WIDTH-1:0] MAP_OUT_AERIN_IDX_r   [0:CORE_H*CORE_W-1];
    reg [MAP_OUT_AER_WIDTH-1:0] MAP_OUT_AERIN_EVENT_r [0:CORE_H*CORE_W-1];

    integer k, l;
    always @(*) begin
        MAP_OUT_AERIN_REQ_r = 'd0;
        for (l = 0; l < LRF_W*LRF_H; l = l + 1) begin
            if (LRF_core_en[l])
                MAP_OUT_AERIN_REQ_r[LRF_core_id[l]] = MAP_IN_AERIN_REQ;
        end
    end
    assign MAP_OUT_AERIN_REQ = ((|event_type) && !(&event_type))? {CORE_H*CORE_W{1'b1}} : MAP_OUT_AERIN_REQ_r;

    localparam int DY_BITS = $clog2(LRF_H);
    localparam int DX_BITS = $clog2(LRF_W);
    localparam int IDX_BITS = C_BITS + DY_BITS + DX_BITS;

    genvar gy, gx, gk;
    generate
        for (gy = 0; gy < LRF_H; gy++) begin
            for (gx = 0; gx < LRF_W; gx++) begin
                localparam int j = gy*LRF_W + gx;
                localparam logic [DY_BITS-1:0] DY = gy;
                localparam logic [DX_BITS-1:0] DX = gx;
                always @(*) begin
                    // 初始化
                    for (k = 0; k < CORE_H*CORE_W; k = k + 1) begin
                        MAP_OUT_AERIN_IDX_r[k]   = {MAP_OUT_AER_WIDTH{1'b0}};
                        MAP_OUT_AERIN_EVENT_r[k] = {2'b11, {MAP_OUT_AER_WIDTH-2{1'b1}}}; // 无效事件
                    end
                    if (LRF_core_en[j]) begin
                        MAP_OUT_AERIN_IDX_r[LRF_core_id[j]] = { in_c, DY, DX };
                        MAP_OUT_AERIN_EVENT_r[LRF_core_id[j]] = {event_type, MAP_OUT_AERIN_IDX_r[LRF_core_id[j]][MAP_OUT_AER_WIDTH-3:0]};
                    end
                end
            end
        end

        for (gk = 0; gk < CORE_H*CORE_W; gk = gk + 1) begin
            assign MAP_OUT_AERIN_IDX[gk]   = ((|event_type) && !(&event_type)) ? {MAP_OUT_AER_WIDTH-2{1'b0}} : MAP_OUT_AERIN_IDX_r[gk];
            assign MAP_OUT_AERIN_EVENT[gk] = ((|event_type) && !(&event_type)) ? map_out_aer_non_neur_event : MAP_OUT_AERIN_EVENT_r[gk];
        end
    endgenerate

    // 所有被命中的核心 ACK 后才 ACK 输入
    assign MAP_IN_AERIN_ACK = &(MAP_OUT_AERIN_ACK | ~MAP_OUT_AERIN_REQ);
    // always @(*) begin
    //     // 初始化
    //     for (k = 0; k < CORE_H*CORE_W; k = k + 1) begin
    //         MAP_OUT_AERIN_IDX_r[k]   = {MAP_OUT_AER_WIDTH{1'b0}};
    //         MAP_OUT_AERIN_EVENT_r[k] = {2'b11, {MAP_OUT_AER_WIDTH-2{1'b1}}}; // 无效事件
    //     end
    //     // 遍历 LRF
    //     for (k = 0; k < LRF_H; k = k + 1) begin
    //         for (l = 0; l < LRF_W*LRF_H; l = l + 1) begin
    //             localparam local_dy = k;
    //             localparam local_dx = l;
    //             if (LRF_core_en[l]) begin
    //                     // IDX: 可自定义线性化地址
    //                 MAP_OUT_AERIN_IDX_r[LRF_core_id[l]] = {in_c, local_dy[j], local_dx[j]};
    //                 MAP_OUT_AERIN_EVENT_r[k] = {event_type, MAP_OUT_AERIN_IDX_r[i][MAP_OUT_AER_WIDTH-3:0]};
    //             end
    //         end
    //     end
    // end
endmodule

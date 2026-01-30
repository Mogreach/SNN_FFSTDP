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
    parameter LRF_H       = 3     // 局部感受野高
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

    localparam RX = LRF_W / 2;
    localparam RY = LRF_H / 2;
    // 局部地址宽度
    localparam DX_W = $clog2(LRF_W);
    localparam DY_W = $clog2(LRF_H);


    // 先算事件能影响的核心坐标范围
    wire [X_BITS-1:0] core_x_min, core_x_max;
    wire [Y_BITS-1:0] core_y_min, core_y_max;

    assign core_x_min = (in_x > RX) ? (in_x - RX) : 0;
    assign core_x_max = (in_x + RX < CORE_W-1) ? (in_x + RX) : CORE_W-1;

    assign core_y_min = (in_y > RY) ? (in_y - RY) : 0;
    assign core_y_max = (in_y + RY < CORE_H-1) ? (in_y + RY) : CORE_H-1;

    // 
    wire [1:0] event_type = MAP_IN_AERIN_EVENT[(MAP_IN_AER_WIDTH-1) -: 2];
    wire [MAP_OUT_AER_WIDTH-1: 0] map_out_aer_non_neur_event = {event_type,{MAP_OUT_AER_WIDTH{1'b1}}};
    wire [MAP_OUT_AER_WIDTH-1: 0] map_out_aer_invalid_event = {2'b11,{MAP_OUT_AER_WIDTH{1'b1}}};

    genvar cx, cy;
    generate
        for (cy = 0; cy < CORE_H; cy = cy + 1) begin : CORE_Y
            for (cx = 0; cx < CORE_W; cx = cx + 1) begin : CORE_X
                localparam CORE_ID = cy * CORE_W + cx;
                localparam CORE_X_COORD = cx + 1;
                localparam CORE_Y_COORD = cy + 1;

                wire hit_x;
                wire hit_y;
 
                // 以输出特征图坐标为中心（cx+1,cy+1），判断输入事件是否落在局部感受野范围内
                assign hit_x = (cx >= core_x_min) && (cx <= core_x_max);
                assign hit_y = (cy >= core_y_min) && (cy <= core_y_max);


                assign MAP_OUT_AERIN_REQ[CORE_ID]  = MAP_IN_AERIN_REQ & hit_x & hit_y;
                // -------------------------------
                // 输入事件相对当前 core 的偏移
                // -------------------------------
                wire signed [$clog2(FM_W):0] dx_signed;
                wire signed [$clog2(FM_H):0] dy_signed;

                assign dx_signed = in_x - cx;
                assign dy_signed = in_y - cy;
                // -------------------------------
                // 核心内部局部地址
                // dx, dy ∈ [0, LRF_W-1], [0, LRF_H-1]
                // -------------------------------
                localparam  DX_W = $clog2(LRF_W);
                localparam  DY_W = $clog2(LRF_H);
                
                wire [DX_W-1:0] local_dx;
                wire [DY_W-1:0] local_dy;
                assign local_dx = dx_signed + RX;
                assign local_dy = dy_signed + RY;

                // assign MAP_OUT_AERIN_IDX[CORE_ID] = {in_c, local_dy, local_dx};
                assign MAP_OUT_AERIN_IDX[CORE_ID] = in_c*local_dy*local_dx + local_dy * local_dx + local_dx;
                assgin MAP_OUT_AERIN_EVENT[CORE_ID] = (|event_type)? map_out_aer_non_neur_event : (hit_x & hit_y) ? {2'b00, MAP_OUT_AERIN_IDX[CORE_ID]} : map_out_aer_invalid_event;
            end
        end
    endgenerate

    // 所有被命中的核心 ACK 后才 ACK 输入
    assign MAP_IN_AERIN_ACK = &(MAP_OUT_AERIN_ACK | ~MAP_OUT_AERIN_REQ);

endmodule

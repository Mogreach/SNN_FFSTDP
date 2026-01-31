module aer_core_event_arbiter #(
    parameter CORE_NUM      = 16,
    parameter AER_OUT_WIDTH = 8  // 不含核心ID位
)(
    input  wire clk,
    input  wire rst,

    // ------------------ 来自每个 core 的输出 ------------------
    input  wire [CORE_NUM-1:0]                    core_req,   // 核心 FIFO 有有效事件
    input  wire [CORE_NUM*AER_OUT_WIDTH-1:0]      core_addr,  // 核心 FIFO 数据
    output wire [CORE_NUM-1:0]                    core_ack,   // 核心 FIFO ACK

    // ------------------ 汇聚输出 ------------------
    output reg                                   evt_req,   // 输出事件请求
    output reg [AER_OUT_WIDTH+$clog2(CORE_NUM)-1:0] evt_addr,  // 输出事件地址
    input  wire                                  evt_ack    // 输出 ACK
);

    localparam CORE_ID_W = $clog2(CORE_NUM);
    localparam EVENT_W   = AER_OUT_WIDTH + CORE_ID_W;

    // ------------------ 特殊事件处理 ------------------
    reg special_event_pending;
    wire special_event_detected;
    integer i;

    // 只要任何核心输出前缀01，就触发一次 special_event
    assign special_event_detected = |(
        {CORE_NUM{1'b0}} // 默认0
        | ({core_addr[CORE_NUM*AER_OUT_WIDTH-1 -: AER_OUT_WIDTH]} & {CORE_NUM{1'b11 << (AER_OUT_WIDTH-2)}}) == 2'b01
    );

    // ------------------ 仲裁器选择普通事件 ------------------
    reg found;
    reg [CORE_ID_W-1:0] sel;

    always @(*) begin
        found = 1'b0;
        sel   = 'd0;
        for (i = 0; i < CORE_NUM; i = i + 1) begin
            if (core_req[i]) begin
                // 检查不是特殊前缀01事件
                if (core_addr[i*AER_OUT_WIDTH +: 2] != 2'b01 && !found) begin
                    found = 1'b1;
                    sel   = i[CORE_ID_W-1:0];
                end
            end
        end
    end

    // ------------------ 输出 REQ/ADDR ------------------
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            evt_req              <= 1'b0;
            evt_addr             <= {EVENT_W{1'b0}};
            special_event_pending <= 1'b0;
        end else begin
            // ---------------- 特殊事件优先 ----------------
            if (special_event_detected && !special_event_pending) begin
                evt_req  <= 1'b1;
                evt_addr <= {2'b01, {AER_OUT_WIDTH+$clog2(CORE_NUM)-2{1'b1}}}; // 输出一次特殊事件
                special_event_pending <= 1'b1;
            end 
            // ---------------- 普通事件 ----------------
            else if (found && !evt_req) begin
                evt_req  <= 1'b1;
                evt_addr <= {core_addr[sel*AER_OUT_WIDTH +: AER_OUT_WIDTH], sel};
            end

            // 输出 ACK 响应
            if (evt_req && evt_ack) begin
                evt_req <= 1'b0;
                if (special_event_pending)
                    special_event_pending <= 1'b0; // 标记清除
            end
        end
    end

    // ------------------ 核心 ACK ------------------
    generate
        genvar j;
        for (j = 0; j < CORE_NUM; j = j + 1) begin : GEN_CORE_ACK
            assign core_ack[j] = evt_req && (sel == j) && evt_ack;
        end
    endgenerate

endmodule

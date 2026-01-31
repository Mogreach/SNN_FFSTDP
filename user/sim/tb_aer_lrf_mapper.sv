`timescale 1ns/1ps

module tb_aer_lrf_mapper;

    // ================= 参数 =================
    localparam FM_C  = 3;
    localparam FM_W  = 8;
    localparam FM_H  = 8;
    localparam CORE_W = 4;
    localparam CORE_H = 4;
    localparam CORE_C = 4;
    localparam LRF_W = 3;
    localparam LRF_H = 3;
    localparam MAP_IN_AER_WIDTH  = 2 + $clog2(FM_C) + $clog2(FM_H) + $clog2(FM_W);
    localparam MAP_OUT_AER_WIDTH = 2 + $clog2(FM_C) + $clog2(LRF_H) + $clog2(LRF_W);
    localparam X_BITS = $clog2(FM_W);
    localparam Y_BITS = $clog2(FM_H);
    localparam C_BITS = $clog2(FM_C);

    // ================= 信号 =================
    logic clk;
    logic rst;

    logic MAP_IN_AERIN_REQ;
    logic [MAP_IN_AER_WIDTH-1:0] MAP_IN_AERIN_EVENT;
    logic [MAP_IN_AER_WIDTH-3:0] MAP_IN_AERIN_IDX;
    wire  MAP_IN_AERIN_ACK;

    wire [CORE_W*CORE_H-1:0] MAP_OUT_AERIN_REQ;
    wire [CORE_W*CORE_H-1:0][MAP_OUT_AER_WIDTH-1:0] MAP_OUT_AERIN_EVENT;
    wire [CORE_W*CORE_H-1:0][MAP_OUT_AER_WIDTH-3:0] MAP_OUT_AERIN_IDX;
    logic [CORE_W*CORE_H-1:0] MAP_OUT_AERIN_ACK;

    // ================= DUT =================
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
    ) dut (
        .clk(clk),
        .rst(rst),
        .MAP_IN_AERIN_REQ(MAP_IN_AERIN_REQ),
        .MAP_IN_AERIN_EVENT(MAP_IN_AERIN_EVENT),
        .MAP_IN_AERIN_IDX(MAP_IN_AERIN_IDX),
        .MAP_IN_AERIN_ACK(MAP_IN_AERIN_ACK),
        .MAP_OUT_AERIN_REQ(MAP_OUT_AERIN_REQ),
        .MAP_OUT_AERIN_EVENT(MAP_OUT_AERIN_EVENT),
        .MAP_OUT_AERIN_IDX(MAP_OUT_AERIN_IDX),
        .MAP_OUT_AERIN_ACK(MAP_OUT_AERIN_ACK)
    );

    // ================= 时钟 =================
    always #5 clk = ~clk;

    // ================= 任务 =================
    task send_event(
        input [1:0] ev_type,
        input [C_BITS-1:0] c,
        input [Y_BITS-1:0] y,
        input [X_BITS-1:0] x
    );
        begin
            @(posedge clk);
            MAP_IN_AERIN_REQ   <= 1'b1;
            MAP_IN_AERIN_EVENT <= {ev_type, {MAP_IN_AER_WIDTH-2{1'b0}}};
            MAP_IN_AERIN_IDX   <= {c, y, x};

            // 等待 ACK
            wait (MAP_IN_AERIN_ACK);
            @(posedge clk);
            MAP_IN_AERIN_REQ <= 1'b0;
        end
    endtask

    // ================= 监视 =================
    integer i;
    always @(posedge clk) begin
        for (i = 0; i < CORE_W*CORE_H; i++) begin
            if (MAP_OUT_AERIN_REQ[i]) begin
                $display("[T=%0t] CORE[%0d] REQ IDX=%h EVENT=%h",
                         $time, i,
                         MAP_OUT_AERIN_IDX[i],
                         MAP_OUT_AERIN_EVENT[i]);
            end
        end
    end

    // ================= 激励 =================
    integer c, y, x;

    initial begin
        clk = 0;
        rst = 1;

        MAP_IN_AERIN_REQ   = 0;
        MAP_IN_AERIN_EVENT = 0;
        MAP_IN_AERIN_IDX   = 0;

        // 所有 core 永远 ready（简化验证）
        MAP_OUT_AERIN_ACK = {CORE_W*CORE_H{1'b1}};

        #20 rst = 0;

        $display("==== Sweep neuron events ====");

        for (c = 0; c < FM_C; c = c + 1) begin
            for (y = 0; y < FM_H; y = y + 1) begin
                for (x = 0; x < FM_W; x = x + 1) begin
                    send_event(2'b00, c[C_BITS-1:0], y[Y_BITS-1:0], x[X_BITS-1:0]);

                    // 防止零时间连续事件（非常重要）
                    @(posedge clk);
                end
            end
        end

        #50;

        $display("==== Send non-neuron event ====");
        send_event(2'b10, 0, 10, 10);

        #50;
        $finish;
    end


endmodule

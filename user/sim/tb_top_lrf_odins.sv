// `timescale 1ns/1ps

module tb_top_lrf_odins();

    // ------------------------------------------------------------
    // Parameters
    // ------------------------------------------------------------
    localparam FM_W    = 16;
    localparam FM_H    = 16;
    localparam FM_C    = 3;
    localparam CORE_W  = 8;
    localparam CORE_H  = 8;
    localparam CORE_C  = 8;
    localparam LRF_W   = 3;
    localparam LRF_H   = 3;
    localparam TIME_STEP = 8;
    localparam POST_NEUR_MEM_WIDTH = 13;
    localparam WEIGHT_WIDTH = 9;
    localparam GOODNESS_WIDTH = 20;

    localparam CLK_PERIOD = 4;
    localparam AERIN_WIDTH = 2 + $clog2(FM_C) + $clog2(FM_H) + $clog2(FM_W);
    localparam AEROUT_WIDTH = 2 + $clog2(CORE_C) + $clog2(CORE_W*CORE_H);


    // ------------------------------------------------------------
    // Signals
    // ------------------------------------------------------------
    logic clk;
    logic rst;
    logic IS_POS;
    logic IS_TRAIN;
    logic ONE_SAMPLE_FINISH;

    logic AERIN_REQ;
    logic [2 + $clog2(FM_C) + $clog2(FM_H) + $clog2(FM_W)-1:0] AERIN_ADDR;
    logic AERIN_ACK;

    logic AEROUT_REQ;
    logic [2 + $clog2(CORE_C) + $clog2(CORE_W*CORE_H) - 1:0] AEROUT_ADDR;
    logic AEROUT_ACK;

    logic [GOODNESS_WIDTH-1:0] GOODNESS;

    // 模拟输出神经元事件
    logic [11:0] aer_neur_spk;
    logic auto_ack_verbose;

    // ------------------------------------------------------------
    // DUT
    // ------------------------------------------------------------
    top_lrf_odins #(
        .FM_W(FM_W),
        .FM_H(FM_H),
        .FM_C(FM_C),
        .CORE_W(CORE_W),
        .CORE_H(CORE_H),
        .CORE_C(CORE_C),
        .LRF_W(LRF_W),
        .LRF_H(LRF_H),
        .TIME_STEP(TIME_STEP),
        .POST_NEUR_MEM_WIDTH(POST_NEUR_MEM_WIDTH),
        .WEIGHT_WIDTH(WEIGHT_WIDTH),
        .GOODNESS_WIDTH(GOODNESS_WIDTH)
    ) dut (
        .clk(clk),
        .rst(rst),
        .IS_POS(IS_POS),
        .IS_TRAIN(IS_TRAIN),
        .ONE_SAMPLE_FINISH(ONE_SAMPLE_FINISH),
        .AERIN_REQ(AERIN_REQ),
        .AERIN_ADDR(AERIN_ADDR),
        .AERIN_ACK(AERIN_ACK),
        .AEROUT_REQ(AEROUT_REQ),
        .AEROUT_ADDR(AEROUT_ADDR),
        .AEROUT_ACK(AEROUT_ACK)
    );

    // ------------------------------------------------------------
    // Clock
    // ------------------------------------------------------------
    always #(CLK_PERIOD/2) clk = ~clk;
    // ------------------------------------------------------------
    // Test stimulus
    // ------------------------------------------------------------
    localparam int FM_HW_BITS = $clog2(FM_C) + $clog2(FM_H) + $clog2(FM_W);
    logic [FM_HW_BITS-1:0] pix_idx;
    initial begin
        clk = 0;
        rst = 1;
        IS_POS = 0;
        IS_TRAIN = 0;
        AERIN_ADDR = 0;
        AERIN_REQ = 0;
        AEROUT_ACK = 0;
        auto_ack_verbose = 1'b1;
        fork
        auto_ack(.req(AEROUT_REQ), .ack(AEROUT_ACK), .addr(AEROUT_ADDR), .neur(aer_neur_spk), .verbose(auto_ack_verbose));
        join_none
        #(2*CLK_PERIOD);
        rst = 0;
        // 模拟几个事件发送到 DUT
        #(3*CLK_PERIOD);

        IS_TRAIN = 1;
        IS_POS = 1;
        wait_ns(20);
        for(int t = 0; t < TIME_STEP; t++) begin
            for (int y = 0; y < 16; y++) begin
                for (int x = 0; x < 16; x++) begin
                    pix_idx = y * FM_W + x;

                    aer_send(
                        {2'b00, pix_idx},   // neuron spike
                        AERIN_ADDR,
                        AERIN_ACK,
                        AERIN_REQ
                    );
                    wait_ns(10);
                end
            end
            aer_send({2'b01, pix_idx}, AERIN_ADDR, AERIN_ACK, AERIN_REQ);
        end
        wait(ONE_SAMPLE_FINISH);
        wait_ns(20000000000);

        $display("GOODNESS: %h", GOODNESS);

        #(10*CLK_PERIOD);
        $display("Simulation completed at time %0t", $time);
        $finish;
    end




    // ------------------------------------------------------------
    // Stimulus task
    // ------------------------------------------------------------
    task automatic aer_send(
        input  logic [AERIN_WIDTH-1:0] addr_in,
        ref    logic [AERIN_WIDTH-1:0] addr_out,
        ref    logic                    ack,
        ref    logic                    req
    );
        while (ack) wait_ns(1);
        addr_out = addr_in;
        wait_ns(5);
        req = 1'b1;
        while (!ack) wait_ns(1);
        wait_ns(5);
        req = 1'b0;
        // 等待 ack 拉低（完整握手）
        while (ack) wait_ns(1);
    endtask


    task automatic auto_ack(
        ref logic req,
        ref logic ack,
        ref logic [AEROUT_WIDTH-1:0] addr,
        ref logic [11:0] neur,
        ref logic verbose
    );
        forever begin
            while (~req) wait_ns(1);
            wait_ns(50);
            neur = addr;
            if (verbose)
                $display("----- NEURON OUTPUT SPIKE (FROM AER): Event 0x%h at time %0t", neur, $time);
            ack = 1'b1;
            while (req) wait_ns(1);
            wait_ns(50);
            ack = 1'b0;
        end
    endtask


    task wait_ns;
        input integer tics_ns;
        #tics_ns;
    endtask
endmodule
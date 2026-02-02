`timescale 1ns/1ps

module tb_aer_core_event_arbiter;

    // -------------------- 参数 --------------------
    localparam CORE_NUM      = 4;
    localparam AER_OUT_CORE_WIDTH = 8;

    localparam CLK_PERIOD = 10;

    // -------------------- 信号 --------------------
    reg  clk;
    reg  rst;

    reg  [CORE_NUM-1:0] core_req;
    reg  [CORE_NUM*AER_OUT_CORE_WIDTH-1:0] core_addr;
    wire [CORE_NUM-1:0] core_ack;

    wire evt_req;
    wire [AER_OUT_CORE_WIDTH+$clog2(CORE_NUM)-1:0] evt_addr;
    reg  evt_ack;

    integer i;

    // -------------------- DUT --------------------
    aer_core_event_arbiter #(
        .CORE_NUM(CORE_NUM),
        .AER_OUT_CORE_WIDTH(AER_OUT_CORE_WIDTH)
    ) dut (
        .clk(clk),
        .rst(rst),
        .core_req(core_req),
        .core_addr(core_addr),
        .core_ack(core_ack),
        .evt_req(evt_req),
        .evt_addr(evt_addr),
        .evt_ack(evt_ack)
    );

    // -------------------- 时钟 --------------------
    initial clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;

    // -------------------- REQ-ACK Task --------------------
    task automatic send_core_event(
        input integer core_idx,
        input [AER_OUT_CORE_WIDTH-1:0] addr
    );
    begin
        // 等待上一事件 ACK 拉低
        wait(core_ack[core_idx]==0);

        // 拉高 core_req
        core_addr[core_idx*AER_OUT_CORE_WIDTH +: AER_OUT_CORE_WIDTH] = addr;
        @(posedge clk);
        core_req[core_idx] = 1'b1;

        // 等待 ACK 拉高
        wait(core_ack[core_idx]==1);
        @(posedge clk);

        // 拉低 core_req
        core_req[core_idx] = 1'b0;
        @(posedge clk);
    end
    endtask

    // -------------------- 自动响应 EVT ACK --------------------
    always @(posedge clk) begin
        if(evt_req) begin
            evt_ack <= 1'b1;
            $display("Time %0t: EVT_REQ=1, EVT_ADDR=0x%h", $time, evt_addr);
        end else begin
            evt_ack <= 1'b0;
        end
    end

    // -------------------- 测试过程 --------------------
    initial begin
        rst = 1;
        core_req = 0;
        core_addr = 0;
        evt_ack = 0;

        #(2*CLK_PERIOD);
        rst = 0;

        $display("==== Test: 普通事件 ====");
        // 依次发送 CORE_NUM 个事件
        for (i=0; i<CORE_NUM; i=i+1) begin
            send_core_event(i, i*8); // 地址 0, 8, 16, 24
        end

        #(2*CLK_PERIOD);

        $display("==== Test: 特殊事件前缀 01 ====");
        // 多核心同时输出前缀 01 的事件
        fork
            send_core_event(0, 8'b0100_0001);
            send_core_event(1, 8'b0101_0010);
            send_core_event(2, 8'b0100_0011);
            send_core_event(3, 8'b0101_0100);
        join

        #(5*CLK_PERIOD);
        $display("==== Test Completed ====");
        $stop;
    end

endmodule

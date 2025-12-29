`timescale 1ns / 1ps

module tb_parallel_to_serial;

    // Parameters
    parameter DATA_WIDTH = 4;
    parameter CNT_MAX = 783;
    parameter STEP = 16;

    // Inputs
    reg CLK;
    reg rst_n;
    reg [DATA_WIDTH-1:0] din_parallel;
    reg din_valid;
    logic AER_IN_ACK;
    logic auto_ack_verbose;
    logic [7:0] aer_neur_spk;
    // Outputs
    reg pts_ready;
    logic [11:0] AER_IN_ADDR;
    logic AER_IN_REQ;
    logic finish;
    // Instantiate the Unit Under Test (UUT)
    parallel_to_serial #(
        .DATA_WIDTH(DATA_WIDTH),
        .CNT_MAX(CNT_MAX),
        .STEP(STEP)
    ) uut (
        .CLK(CLK),
        .rst_n(rst_n),
        .din_parallel(din_parallel),
        .din_valid(din_valid),
        .AER_IN_ACK(AER_IN_ACK),
        .pts_ready(pts_ready),
        .AER_IN_ADDR(AER_IN_ADDR),
        .AER_IN_REQ(AER_IN_REQ),
        .finish(finish)
    );

    // Clock generation
    initial begin
        CLK = 1;
        forever #5 CLK = ~CLK; // 100MHz clock
    end

    // Test stimulus
    initial begin
        // Initialize Inputs
        auto_ack_verbose = 1'b1;
        fork
        auto_ack(.req(AER_IN_REQ), .ack(AER_IN_ACK), .addr(AER_IN_ADDR), .neur(aer_neur_spk), .verbose(auto_ack_verbose));
        join_none
        rst_n = 0;
        din_parallel = 0;
        din_valid = 0;
        AER_IN_ACK = 0;

        // Reset the design
        #20;
        rst_n = 1;
        for (int i = 0; i < 3136; i = i + 1) begin
            din_parallel <= i;
            wait_ns(10);
            din_valid <= 1;
            wait_ns(10);
            din_valid <= 0;
            wait_ns(10);
            while(pts_ready == 0)begin
                wait_ns(10);
            end
        end
        $stop;
    end
task automatic auto_ack (
        ref    logic       req,
        ref    logic       ack,
        ref    logic [11:0] addr,
        ref    logic [11:0] neur,
        ref    logic       verbose
    );
    
        forever begin
            while (~req) wait_ns(1);
            wait_ns(100);
            neur = addr;
            if (verbose)
                $display("----- NEURON OUTPUT SPIKE (FROM AER): Event from neuron %d", neur);
            ack = 1'b1;
            while (req) wait_ns(1);
            wait_ns(100);
            ack = 1'b0;
        end
endtask
task wait_ns;
    input   tics_ns;
    integer tics_ns;
    #tics_ns;
endtask
endmodule
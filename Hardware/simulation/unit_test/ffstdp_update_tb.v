`timescale 1ns / 1ps

module tb_ffstdp_update;

    // Parameters
    parameter PRE_CNT_WIDTH = 8;
    parameter POST_CNT_WIDTH = 7;
    parameter WEIGHT_WIDTH = 8;

    // Inputs
    reg CLK;
    reg CTRL_TREF_EVENT;
    reg IS_POS;
    reg IS_TRAIN;
    reg [POST_CNT_WIDTH-1:0] POST_SPIKE_CNT;
    reg [PRE_CNT_WIDTH-1:0] PRE_SPIKE_CNT;
    reg signed [WEIGHT_WIDTH-1:0] WSYN_CURR;

    // Outputs
    wire signed [WEIGHT_WIDTH-1:0] WSYN_NEW;
    wire [WEIGHT_WIDTH-1:0] L_to_s_derivative;
    wire [WEIGHT_WIDTH-1:0] L_to_s_derivative_pos;
    wire [WEIGHT_WIDTH-1:0] L_to_s_derivative_neg;
    wire [PRE_CNT_WIDTH + WEIGHT_WIDTH-1:0] L_to_w_derivative;
    wire [WEIGHT_WIDTH-1:0] delta_w;
    wire signed [WEIGHT_WIDTH-1:0] delta_w_signed;
    wire signed [WEIGHT_WIDTH-1:0] new_w_result;
    wire overflow;

    // Instantiate the sdsp_update module
    ffstdp_update #(
        .PRE_CNT_WIDTH(PRE_CNT_WIDTH),
        .POST_CNT_WIDTH(POST_CNT_WIDTH),
        .WEIGHT_WIDTH(WEIGHT_WIDTH)
    ) uut (
        .CTRL_TREF_EVENT(CTRL_TREF_EVENT),
        .IS_POS(IS_POS),
        .IS_TRAIN(IS_TRAIN),
        .POST_SPIKE_CNT(POST_SPIKE_CNT),
        .PRE_SPIKE_CNT(PRE_SPIKE_CNT),
        .WSYN_CURR(WSYN_CURR),
        .WSYN_NEW(WSYN_NEW)
    );

    // Clock generation
    always begin
        #5 CLK = ~CLK;  // Generate a clock with a period of 10ns
    end
    
	// always @ ( posedge CLK ) begin
	//     POST_SPIKE_CNT <= POST_SPIKE_CNT + 'd1;
	// end

    // Stimulus block
    initial begin
        // Initialize inputs
        IS_TRAIN = 1; // Enable training mode
        CLK = 0;
        CTRL_TREF_EVENT = 1;
        IS_POS = 1;
        POST_SPIKE_CNT = 7'd0;
        PRE_SPIKE_CNT = 8'd0;
        WSYN_CURR = 8'd50; // Example initial weight
        #20;
        // POS情况
        for (int n = 0; n < 16; n++) begin
           PRE_SPIKE_CNT = n; // Example pre-spike count
           for (int m = 0; m < 16; m++) begin
               POST_SPIKE_CNT = m; // Example post-spike count
               WSYN_CURR = $urandom_range(0, 255); // 8-bit random weight
               #10; // Wait for a while to observe the output
           end
        end 
        // NEG情况
        IS_POS = 0; // Change to negative case
        for (int n = 0; n < 16; n++) begin
           PRE_SPIKE_CNT = n; // Example pre-spike count
           for (int m = 0; m < 16; m++) begin
               POST_SPIKE_CNT = m; // Example post-spike count
               WSYN_CURR = $urandom_range(0, 255); // 8-bit random weight
               #10; // Wait for a while to observe the output
           end
        end 
        // End simulation
    end

    // Monitor the output
    initial begin
        $monitor("Time = %0t | CTRL_TREF_EVENT = %b | IS_POS = %b | POST_SPIKE_CNT = %d | PRE_SPIKE_CNT = %d | WSYN_CURR = %d | WSYN_NEW = %d | L_to_s_derivative = %d | delta_w = %d | delta_w_signed = %d | new_w_result = %d | overflow = %b", 
                 $time, CTRL_TREF_EVENT, IS_POS, POST_SPIKE_CNT, PRE_SPIKE_CNT, WSYN_CURR, WSYN_NEW, L_to_s_derivative, delta_w, delta_w_signed, new_w_result, overflow);
    end

endmodule

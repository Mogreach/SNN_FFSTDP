// weight = lr * (PRE_SPIKE_CNT - POST_SPIKE_CNT) * AVG_GOODNESS
module ffstdp_update #(
    parameter PRE_CNT_WIDTH = 8, //计数0-31
    parameter POST_CNT_WIDTH = 7,
    parameter WEIGHT_WIDTH = 8,
    parameter GRAD_WIDTH = 8
)(
    // Inputs
    input wire              AVG_GOODNESS,
    // General
    input  wire             CLK, 
    input  wire             CTRL_TREF_EVENT,
    input  wire             IS_POS,   
    input  wire             IS_TRAIN,
    // From neuron 
    input  wire [POST_CNT_WIDTH-1:0]       POST_SPIKE_CNT,
    input  wire [PRE_CNT_WIDTH-1:0]        PRE_SPIKE_CNT, 
    // From SRAM
    input wire signed [WEIGHT_WIDTH-1:0] WSYN_CURR,
    input wire signed [GRAD_WIDTH-1:0] GRAD_CURR,
	// Output
	output reg signed [WEIGHT_WIDTH-1:0] WSYN_NEW,
    output reg signed [GRAD_WIDTH-1:0] GRAD_NEW
);

    localparam weight_max_value = (1 << (WEIGHT_WIDTH-1)) - 1;
    localparam weight_min_value = -(1 << (WEIGHT_WIDTH-1));
    localparam grad_max_value = (1 << (GRAD_WIDTH-1)) - 1;
    localparam grad_min_value = -(1 << (GRAD_WIDTH-1));
    localparam scale = 4;

    // ReLU * spike item
    wire [$clog2(POST_CNT_WIDTH):0] pre_spike_valid_sum;
    adder_tree #(
        .N     	(POST_CNT_WIDTH),
        .WIDTH 	(1)
    )
    u_adder_tree(
        .in  	(PRE_SPIKE_CNT & POST_SPIKE_CNT),
        .out 	(pre_spike_valid_sum)
    );

    //------------CLK 1--------------------
    // Positive derivative
    wire signed [GRAD_WIDTH-1:0] L_to_s_derivative_pos = pre_spike_valid_sum * AVG_GOODNESS;
    // Negative derivative
    wire signed [GRAD_WIDTH-1:0] L_to_s_derivative_neg = pre_spike_valid_sum * (1'b1 - AVG_GOODNESS);
    // SRAM read delay registers
    reg signed [WEIGHT_WIDTH-1:0] WSYN_CURR_reg;
    reg signed [GRAD_WIDTH-1:0] GRAD_CURR_reg;
    always @(posedge CLK)           
    begin                                        
        WSYN_CURR_reg <= WSYN_CURR;
        GRAD_CURR_reg <= GRAD_CURR;                        
    end 
    //------------CLK 2--------------------
    // Select derivative based on IS_POS
    wire signed [GRAD_WIDTH-1:0] L_to_s_derivative = (IS_POS)? L_to_s_derivative_pos : L_to_s_derivative_neg;
    // Gradient accumulation
    wire signed [GRAD_WIDTH-1:0] GRAD_ACCUM = GRAD_CURR_reg + (L_to_s_derivative >>> scale);
    // New weight result
    wire signed [WEIGHT_WIDTH-1:0] new_w_result = WSYN_CURR_reg + GRAD_ACCUM;
    // Overflow detection for weight and gradient
    wire weight_overflow = !(WSYN_CURR_reg[WEIGHT_WIDTH-1] ^ GRAD_ACCUM[GRAD_WIDTH-1]) && (new_w_result[WEIGHT_WIDTH-1] ^ WSYN_CURR_reg[WEIGHT_WIDTH-1]);
    wire grad_overflow = !(GRAD_CURR_reg[GRAD_WIDTH-1] ^ L_to_s_derivative[GRAD_WIDTH-1]) && (GRAD_ACCUM[GRAD_WIDTH-1] ^ GRAD_CURR_reg[GRAD_WIDTH-1]);

    // Select final weight and gradient values considering overflow
    wire signed [WEIGHT_WIDTH-1:0] weight_slect =  weight_overflow? 
                                            (new_w_result[WEIGHT_WIDTH-1] == 1'b1)? weight_max_value : weight_min_value 
                                            : new_w_result;   
    wire signed [GRAD_WIDTH-1:0] grad_slect =  grad_overflow? 
                                            (GRAD_ACCUM[GRAD_WIDTH-1] == 1'b1)? grad_max_value : grad_min_value 
                                            : GRAD_ACCUM;

                                         
	always @(*) begin
        // Update weight during training and enable training
		if (CTRL_TREF_EVENT && IS_TRAIN) begin 
            WSYN_NEW = weight_slect;
        end
        else begin
            WSYN_NEW = WSYN_CURR_reg;
        end
        // Update gradient only during inference and disable training; Once updating weight, gradient is set to zero
        if (CTRL_TREF_EVENT && !IS_TRAIN) begin
            GRAD_NEW = grad_slect;
        end
        else if (CTRL_TREF_EVENT && IS_TRAIN) begin
            GRAD_NEW = 'd0;
        end
        else begin
            GRAD_NEW = GRAD_CURR_reg;
        end
	end 
    
endmodule

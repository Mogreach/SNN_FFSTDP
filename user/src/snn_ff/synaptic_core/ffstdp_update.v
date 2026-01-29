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

    localparam max_value = (1 << (WEIGHT_WIDTH-1)) - 1;
    localparam min_value = -(1 << (WEIGHT_WIDTH-1));

    reg signed [WEIGHT_WIDTH-1:0] WSYN_CURR_reg;
    reg signed [GRAD_WIDTH-1:0] GRAD_CURR_reg;
    reg signed [WEIGHT_WIDTH-1:0] delta_w_signed_reg;


    wire [7:0] rom_address;
    wire [POST_CNT_WIDTH-1:0]       POST_SPIKE_CNT_encoded;
    wire [PRE_CNT_WIDTH-1:0]        PRE_SPIKE_CNT_encoded;
    wire POST_SPIKE_CNT_eq_zero;
    wire PRE_SPIKE_CNT_eq_zero;

    wire signed [WEIGHT_WIDTH-1:0] L_to_s_derivative_pos; //Q0.8
    wire signed [WEIGHT_WIDTH-1:0] L_to_s_derivative_neg; //Q0.8
    wire signed [WEIGHT_WIDTH-1:0] L_to_s_derivative; //Q1.7

    wire signed [WEIGHT_WIDTH-1:0] delta_w_signed; //Q3.4
    wire signed [WEIGHT_WIDTH-1:0] new_w_result;   //Q3.4


    assign PRE_SPIKE_CNT_encoded = PRE_SPIKE_CNT - 1'b1;
    assign POST_SPIKE_CNT_encoded = POST_SPIKE_CNT - 1'b1;
    assign rom_address =  {PRE_SPIKE_CNT_encoded[3:0],POST_SPIKE_CNT_encoded[3:0]};

    assign PRE_SPIKE_CNT_eq_zero = !(|PRE_SPIKE_CNT);
    assign POST_SPIKE_CNT_eq_zero = !(|POST_SPIKE_CNT);
    assign L_to_s_derivative = (IS_POS)? L_to_s_derivative_pos : L_to_s_derivative_neg;
    assign delta_w_signed = (PRE_SPIKE_CNT_eq_zero || POST_SPIKE_CNT_eq_zero)? 'd0 : (L_to_s_derivative);
    assign new_w_result = WSYN_CURR_reg + delta_w_signed_reg;// 权重值同为Q3.4，其中1位符号位，3位整数位，4位小数位，

    assign overflow = !(WSYN_CURR_reg[WEIGHT_WIDTH-1] ^ delta_w_signed_reg[WEIGHT_WIDTH-1]) && (new_w_result[WEIGHT_WIDTH-1] ^ WSYN_CURR_reg[WEIGHT_WIDTH-1]);
    always @(posedge CLK)           
    begin                                        
        WSYN_CURR_reg <= WSYN_CURR;
        GRAD_CURR_reg <= GRAD_CURR;
        delta_w_signed_reg <= delta_w_signed;                          
    end 
    wire signed [WEIGHT_WIDTH-1:0] w_slect =  overflow? 
                                            (new_w_result[WEIGHT_WIDTH-1] == 1'b1)? max_value : min_value 
                                            : new_w_result;                                        
	always @(*) begin
		if (CTRL_TREF_EVENT && IS_TRAIN) begin 
            WSYN_NEW = w_slect;
            GRAD_NEW = 1'b1;
        end
        else begin
            WSYN_NEW = WSYN_CURR_reg;
            GRAD_NEW = GRAD_CURR_reg;
        end   
	end 
    
    // pos_derivative_rom pos_derivative_rom_0(
    // .a(rom_address),      // input wire [7 : 0] a
    // .spo(L_to_s_derivative_pos)  // output wire [7 : 0] spo
    // );
    // neg_derivative_rom neg_derivative_rom_0(
    // .a(rom_address),      // input wire [7 : 0] a
    // .spo(L_to_s_derivative_neg)  // output wire [7 : 0] spo
    // );
    pos_derivative_rom u_pos_derivative_rom
    (
        .clk                                   (CLK                ),
        .addr                                  (rom_address        ),
        .dout                                  (L_to_s_derivative_pos) 
    );
    neg_derivative_rom u_neg_derivative_rom
    (
        .clk                                   (CLK                ),
        .addr                                  (rom_address        ),
        .dout                                  (L_to_s_derivative_neg) 
    );

endmodule

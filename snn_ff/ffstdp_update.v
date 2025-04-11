module ffstdp_update #(
    parameter PRE_CNT_WIDTH = 8, //计数0-31
    parameter POST_CNT_WIDTH = 7,
    parameter WEIGHT_WIDTH = 8
)(
    // Inputs
    // General
    input  wire             CTRL_TREF_EVENT,
    input  wire             IS_POS,   
    input  wire             IS_TRAIN,
    // From neuron 
    input  wire [POST_CNT_WIDTH-1:0]       POST_SPIKE_CNT,
    input  wire [PRE_CNT_WIDTH-1:0]        PRE_SPIKE_CNT, 
    // From SRAM
    input wire signed [WEIGHT_WIDTH-1:0] WSYN_CURR,
	// Output
	output reg signed [WEIGHT_WIDTH-1:0] WSYN_NEW
);
    localparam pre_cnt_actual_width = 5;
    localparam lr_rate_scale_factor = 4;
    localparam L_to_w_d_width = pre_cnt_actual_width + WEIGHT_WIDTH-1; 
    localparam max_value = (1 << (WEIGHT_WIDTH-1)) - 1;
    localparam min_value = -(1 << (WEIGHT_WIDTH-1));
    wire pos_rom_en;
    wire neg_rom_en;

    wire [WEIGHT_WIDTH-1:0] L_to_s_derivative; //Q1.7
    wire [WEIGHT_WIDTH-1:0] L_to_s_derivative_pos; //Q0.8
    wire [WEIGHT_WIDTH-1:0] L_to_s_derivative_neg; //Q0.8
    wire [L_to_w_d_width:0] L_to_w_derivative;
    wire [WEIGHT_WIDTH-1:0] delta_w;
    wire signed [WEIGHT_WIDTH-1:0] delta_w_signed;
    wire signed [WEIGHT_WIDTH-1:0] new_w_result;

    assign pos_rom_en = CTRL_TREF_EVENT && IS_POS;
    assign neg_rom_en = CTRL_TREF_EVENT && !IS_POS;
    assign L_to_s_derivative = IS_POS? L_to_s_derivative_pos : L_to_s_derivative_neg;//2*freq*L_dervative：由Q0.8变为Q1.7相当于乘于2
    assign L_to_w_derivative = (L_to_s_derivative * PRE_SPIKE_CNT[pre_cnt_actual_width-1:0]);//相当于Q5.0 * Q1.7得到Q6.7的结果
    // assign delta_w = {2'b0 , L_to_w_derivative[L_to_w_d_width : L_to_w_d_width-lr_rate_scale_factor+1]};//对Q6.7放缩2^-6转为Q2.6
    assign delta_w = (L_to_w_derivative[L_to_w_d_width : L_to_w_d_width - WEIGHT_WIDTH + 1]);//对Q6.7截取高8位后进行放缩
    // assign delta_w_signed[WEIGHT_WIDTH-1] = !IS_POS;//delta_w_signed符号位，正样本为正，负样本为负
    assign delta_w_signed = IS_POS? {1'b0,delta_w[6:0]} : {1'b1,~delta_w[6:0]} + 1'b1;//转为补码

    assign new_w_result = WSYN_CURR + delta_w_signed;// 权重值同为Q2.6，其中1位符号位，1位整数位，6位小数位，
    assign overflow = (WSYN_CURR[WEIGHT_WIDTH-1]==delta_w_signed[WEIGHT_WIDTH-1]) && (new_w_result[WEIGHT_WIDTH-1]!=WSYN_CURR[WEIGHT_WIDTH-1]);

	always @(*) begin
		if      (CTRL_TREF_EVENT && IS_TRAIN) WSYN_NEW = overflow? 
                                            (new_w_result[WEIGHT_WIDTH-1] == 1'b1)? max_value : min_value 
                                            : new_w_result;
		else    WSYN_NEW = WSYN_CURR;
	end 

    pos_derivative_rom pos_derivative_rom_0(
    .a(POST_SPIKE_CNT[4:0]),  // input wire [4 : 0] a
    .spo(L_to_s_derivative_pos)  // output wire [7 : 0] spo
    );
    neg_derivative_rom neg_derivative_rom_0(
    .a(POST_SPIKE_CNT[4:0]),  // input wire [4 : 0] a
    .spo(L_to_s_derivative_neg)  // output wire [7 : 0] spo
    );

endmodule

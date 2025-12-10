module adder_tree #(
    parameter N = 4,                // 输入数量
    parameter WIDTH = 16            // 输入位宽
)(
    input  wire [WIDTH*N-1:0] in,  // 输入数组
    output wire [WIDTH+$clog2(N)-1:0] out  // 输出为扩展位宽
);
    wire[WIDTH-1 : 0] in_group [0:N-1];
    genvar idx;
    // 将输入展成数组形式
    generate
        for (idx = 0; idx < N; idx = idx + 1) begin : input_array
            assign in_group[idx] = in[WIDTH*(idx+1)-1 : WIDTH*idx];
        end
    endgenerate
    // 如果 N = 1，直接输出
    generate
        if (N == 1) begin : base_case
            assign out = in_group[0];
        end else begin : recursive_case
            // 上一层的节点数
            localparam NEXT_N = (N+1)/2;
            localparam NEXT_WIDTH = WIDTH+1;

            // 下一层输入
            wire [NEXT_WIDTH-1:0] next_in [0:NEXT_N-1];

            genvar i;
            for (i = 0; i < NEXT_N; i = i + 1) begin
                // 成对相加
                assign next_in[i] = in_group[2*i] + in_group[2*i+1];
            end
            // 下一层输入数组转成总线形式
            wire [NEXT_WIDTH*NEXT_N - 1: 0] next_in_bus;
            for (idx = 0; idx < NEXT_N; idx = idx + 1) begin : input_array
                assign next_in_bus[NEXT_WIDTH*idx +: NEXT_WIDTH] = next_in[idx];
            end
            // 递归实例化
            adder_tree #(
                .N(NEXT_N),
                .WIDTH(NEXT_WIDTH)
            ) next_level (
                .in(next_in_bus),
                .out(out)
            );
        end
    endgenerate

endmodule

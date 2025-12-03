module adder_tree #(
    parameter NUM = 4,                 // number of inputs
    parameter IN_WIDTH = 10            // width of each input
)(
    input  wire [IN_WIDTH-1:0] in [0:NUM-1],
    output wire [IN_WIDTH+$clog2(NUM)-1:0] sum
);

    localparam OUT_WIDTH = IN_WIDTH + $clog2(NUM);
    localparam LEVELS = $clog2(NUM);

    // 多维数组：layer[level][node]
    wire [OUT_WIDTH-1:0] layer [0:LEVELS][0:NUM-1];

    genvar i, l;

    // 第 0 层 = 直接输入（位宽扩展）
    generate
        for (i = 0; i < NUM; i = i + 1) begin : INIT_LAYER
            assign layer[0][i] = in[i];
        end
    endgenerate

    // 构建加法树
    generate
        for (l = 1; l <= LEVELS; l = l + 1) begin : LEVEL_LOOP
            localparam PREV_NUM = (NUM + (1<<(l-1)) - 1) >> (l-1); // 上层节点数
            localparam CURR_NUM = (NUM + (1<<l) - 1) >> l;         // 当前层节点数

            for (i = 0; i < CURR_NUM; i = i + 1) begin : ADD_NODE
                if (2*i + 1 < PREV_NUM) begin
                    assign layer[l][i] = layer[l-1][2*i] + layer[l-1][2*i+1];
                end else begin
                    assign layer[l][i] = layer[l-1][2*i]; // 透传（单节点）
                end
            end
        end
    endgenerate

    // 最终输出
    assign sum = layer[LEVELS][0];

endmodule

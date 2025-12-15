module sram_synaptic_bank #(
    parameter DATA_WIDTH = 32,
    parameter TOTAL_DEPTH = 12544,
    parameter BLOCK_DEPTH = 2048
)(
    input CK,
    input CS,
    input WE,
    input [$clog2(TOTAL_DEPTH)-1:0] A,
    input [DATA_WIDTH-1:0] D,
    output [DATA_WIDTH-1:0] Q
);

    localparam BANK_NUM = (TOTAL_DEPTH + BLOCK_DEPTH - 1) / BLOCK_DEPTH;
    localparam BANK_ADDR_W = $clog2(BLOCK_DEPTH);
    localparam BANK_ID_W   = $clog2(BANK_NUM);

    wire [BANK_ID_W-1:0] bank_id   = A / BLOCK_DEPTH;
    wire [BANK_ADDR_W-1:0] bank_addr = A % BLOCK_DEPTH;

    wire [DATA_WIDTH-1:0] bank_q [0:BANK_NUM-1];

    genvar i;
    generate
        for (i = 0; i < BANK_NUM; i = i + 1) begin : bank_gen
            sram_bank #(
                .DATA_WIDTH (DATA_WIDTH),
                .BANK_ID    (i),
                .TOTAL_DEPTH(TOTAL_DEPTH),
                .BLOCK_DEPTH(BLOCK_DEPTH)
            ) bank (
                .CK (CK),
                .CS (CS & (bank_id == i)),
                .WE (WE & (bank_id == i)),
                .A  (bank_addr),
                .D  (D),
                .Q  (bank_q[i])
            );
        end
    endgenerate

    // Select correct bank output
    assign Q = bank_q[bank_id];

endmodule
module sram_bank #(
    parameter DATA_WIDTH = 32,
    parameter BANK_ID = 0,
    parameter TOTAL_DEPTH = 0,
    parameter BLOCK_DEPTH = 2048
)(
    input CK,
    input CS,
    input WE,
    input [$clog2(BLOCK_DEPTH)-1:0] A,
    input [DATA_WIDTH-1:0] D,
    output reg [DATA_WIDTH-1:0] Q
);

    reg [DATA_WIDTH-1:0] mem [0:BLOCK_DEPTH-1];

    integer i;
    integer global_index;
    reg [DATA_WIDTH-1:0] temp_mem [0:TOTAL_DEPTH-1];

    // 自动从总文件中加载属于自己的范围
    initial begin
        $readmemh("D:/OneDrive/SNN_FFSTDP/Gen_out/weights_weight.txt", temp_mem);

        for (i = 0; i < BLOCK_DEPTH; i = i + 1) begin
            global_index = BANK_ID * BLOCK_DEPTH + i;
            if (global_index < TOTAL_DEPTH)
                mem[i] = temp_mem[global_index];
            else
                mem[i] = 0;
        end
    end

    always @(posedge CK) begin
        if (CS) begin
            Q <= mem[A];
            if (WE) mem[A] <= D;
        end
    end

endmodule

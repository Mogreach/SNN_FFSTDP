module goodness_moving_avg #(
    parameter CORE_NUM              = 4,
    parameter POST_NEUR_PARALLEL     = 8,
    parameter POST_NEUR_MEM_WIDTH    = 13,
    parameter GOODNESS_WIDTH        = 20,
    parameter AVG_SHIFT              = 4   // 1/16 EMA
)(
    input  wire                               clk,
    input  wire                               rst,

    input  wire [CORE_NUM-1:0]                core_valid,
    input  wire [CORE_NUM-1:0]                core_clear_goodness,
    input  wire [CORE_NUM * POST_NEUR_PARALLEL * POST_NEUR_MEM_WIDTH-1:0] core_mem_bus,

    output wire  [CORE_NUM * GOODNESS_WIDTH - 1 :0] avg_mem_bus
);
genvar c, n;
generate
    for (c = 0; c < CORE_NUM; c = c + 1) begin: gen_mem_adder_each_core
        // mem bus of each core
        wire [POST_NEUR_PARALLEL* POST_NEUR_MEM_WIDTH-1:0] mem_bus_each_core;
        assign mem_bus_each_core = core_mem_bus[c*POST_NEUR_PARALLEL*POST_NEUR_MEM_WIDTH +: POST_NEUR_PARALLEL*POST_NEUR_MEM_WIDTH];
        // sign bit bus of each core
        wire [POST_NEUR_PARALLEL-1:0] mem_sign_bus_each_core; 
        // ReLU mem bus of each core
        wire [POST_NEUR_PARALLEL * (POST_NEUR_MEM_WIDTH-1) -1:0] mem_relu_bus_each_core;
        // Assign sign bit and ReLU mem bus
        for (n = 0; n < POST_NEUR_PARALLEL; n = n + 1) begin: gen_sign_bit_each_neur
            wire [POST_NEUR_MEM_WIDTH-1:0] mem_bus_each_neur;
            assign mem_bus_each_neur = mem_bus_each_core[n*POST_NEUR_MEM_WIDTH +: POST_NEUR_MEM_WIDTH];
            assign mem_sign_bus_each_core[n] = mem_bus_each_neur[POST_NEUR_MEM_WIDTH-1];
            assign mem_relu_bus_each_core[n*(POST_NEUR_MEM_WIDTH-1) +: POST_NEUR_MEM_WIDTH-1] = mem_sign_bus_each_core[n] ? 'd0 : mem_bus_each_neur[POST_NEUR_MEM_WIDTH-2:0]; 
        end
        // Adder tree to sum ReLU mems
        wire [POST_NEUR_MEM_WIDTH-1 + (POST_NEUR_PARALLEL)-1:0] mem_sum_each_core;
        adder_tree #(
            .N     	(POST_NEUR_PARALLEL),
            .WIDTH 	(POST_NEUR_MEM_WIDTH-1)
        )
        u_adder_tree(
            .in  	( mem_relu_bus_each_core),
            .out 	( mem_sum_each_core)
        );
        // Average calculation
        wire [POST_NEUR_MEM_WIDTH - 1 - 1 :0] mem_avg_each_core =  mem_sum_each_core >>> POST_NEUR_PARALLEL;
        
        // Register delay
        reg [POST_NEUR_MEM_WIDTH - 1 - 1 :0] mem_avg_each_core_reg;
        reg mem_valid_reg;
        always @(posedge clk) begin
            if (rst) begin
                mem_avg_each_core_reg <= 'd0;
                mem_valid_reg <= 1'b0;
            end
            else begin
                mem_avg_each_core_reg <= mem_avg_each_core;
                mem_valid_reg <= core_valid[c];  
            end
        end

        // EMA update
        reg [GOODNESS_WIDTH-1 : 0] move_avg_mem;
        always @(posedge clk or posedge rst) begin
            if (rst) begin
                move_avg_mem <= 'd0;
            end
            else if (core_clear_goodness[c]) begin
                move_avg_mem <= 'd0;
            end
            else if (mem_valid_reg) begin
                move_avg_mem <= move_avg_mem - (move_avg_mem >>> AVG_SHIFT) + (mem_avg_each_core_reg >>> AVG_SHIFT);
            end
            else begin
                move_avg_mem <= move_avg_mem;
            end
        end

        // Assign to output bus
        assign avg_mem_bus[c*GOODNESS_WIDTH +: GOODNESS_WIDTH] = move_avg_mem;
    end
endgenerate
endmodule
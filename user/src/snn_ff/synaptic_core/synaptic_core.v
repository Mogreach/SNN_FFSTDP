module synaptic_core #(
    parameter TIME_STEP = 8,
    parameter INPUT_NEURON = 784,
    parameter OUTPUT_NEURON = 256,
    parameter AER_WIDTH = 12,

    parameter PRE_NEUR_ADDR_WIDTH = 10,
    parameter PRE_NEUR_WORD_ADDR_WIDTH = 10,
    parameter PRE_NEUR_BYTE_ADDR_WIDTH = 0,

    parameter POST_NEUR_ADDR_WIDTH = 10,
    parameter POST_NEUR_WORD_ADDR_WIDTH = 8,
    parameter POST_NEUR_BYTE_ADDR_WIDTH = 2,
    parameter POST_NEUR_PARALLEL = 4,

    parameter PRE_NEUR_DATA_WIDTH = 8,
    parameter POST_NEUR_DATA_WIDTH = 32,
    parameter POST_NEUR_MEM_WIDTH = 12,
    parameter POST_NEUR_SPIKE_CNT_WIDTH = 7,

    parameter SYN_ARRAY_DATA_WIDTH = 32,
    parameter SYN_ARRAY_ADDR_WIDTH = 16,

    parameter GRAD_ARRAY_DATA_WIDTH = 32,
    parameter GRAD_ARRAY_ADDR_WIDTH = 16,

    parameter WEIGHT_WIDTH = 8,
    parameter GRAD_WIDTH = 8,
    parameter GOODNESS_WIDTH = 20
) (
    input wire IS_POS,
    input wire IS_TRAIN,
    // Global inputs ------------------------------------------
    input wire CLK,
    input  wire        [              15:0] AVG_GOODNESS,
    // Inputs from controller ---------------------------------
    input wire                            CTRL_SYNARRAY_CS,
    input wire                            CTRL_SYNARRAY_WE,
    input wire [SYN_ARRAY_ADDR_WIDTH-1:0] CTRL_SYNARRAY_ADDR,
    input wire [POST_NEUR_ADDR_WIDTH-1:0] CTRL_POST_NEURON_ADDRESS,

    input wire CTRL_GRAD_ARRAY_CS,
    input wire CTRL_GRAD_ARRAY_WE,


    input  wire                                                         CTRL_NEUR_EVENT,
    input  wire                                                         CTRL_TSTEP_EVENT,
    input  wire                                                         CTRL_TREF_EVENT,
    // Inputs from neurons ------------------------------------
    input  wire [                              PRE_NEUR_DATA_WIDTH-1:0] PRE_NEUR_S_CNT,
    input  wire [POST_NEUR_SPIKE_CNT_WIDTH * POST_NEUR_PARALLEL -1 : 0] POST_NEUR_S_CNT,
    // Outputs ------------------------------------------------
    output wire [                             SYN_ARRAY_DATA_WIDTH-1:0] SYNARRAY_RDATA
);
    // Internal regs and wires definitions
    wire [     SYN_ARRAY_ADDR_WIDTH-1:0] synarray_addr;
    wire [     SYN_ARRAY_DATA_WIDTH-1:0] synarray_wdata;
    wire [    GRAD_ARRAY_ADDR_WIDTH-1:0] grad_array_addr;
    wire [    GRAD_ARRAY_DATA_WIDTH-1:0] grad_array_wdata;
    wire [    GRAD_ARRAY_DATA_WIDTH-1:0] GRAD_ARRAY_RDATA;

    wire [POST_NEUR_BYTE_ADDR_WIDTH-1:0] post_neuron_byte_addr;

    wire [             WEIGHT_WIDTH-1:0] weight_orignal_array  [0:POST_NEUR_PARALLEL-1];
    wire [             WEIGHT_WIDTH-1:0] weight_new_array      [0:POST_NEUR_PARALLEL-1];
    wire [             GRAD_WIDTH-1:0] weight_grad_orignal_array  [0:POST_NEUR_PARALLEL-1];
    wire [             GRAD_WIDTH-1:0] weight_grad_new_array      [0:POST_NEUR_PARALLEL-1];

    wire [POST_NEUR_SPIKE_CNT_WIDTH-1:0] POST_NEUR_S_CNT_array [0:POST_NEUR_PARALLEL-1];

    assign synarray_addr = CTRL_SYNARRAY_ADDR;

    genvar i;
    generate
        for (i = 0; i < POST_NEUR_PARALLEL; i = i + 1) begin : gen_ffstdp_update
            wire [WEIGHT_WIDTH-1:0] weight_orignal = SYNARRAY_RDATA[i*WEIGHT_WIDTH+:WEIGHT_WIDTH];
            wire [WEIGHT_WIDTH-1:0] weight_new;
            wire [GRAD_WIDTH-1:0] weight_gradient_original = GRAD_ARRAY_RDATA[i*GRAD_WIDTH+:GRAD_WIDTH];
            wire [GRAD_WIDTH-1:0] weight_gradient_new;

            assign POST_NEUR_S_CNT_array[i] = POST_NEUR_S_CNT[i*POST_NEUR_SPIKE_CNT_WIDTH+:POST_NEUR_SPIKE_CNT_WIDTH];
            assign weight_orignal_array[i] = weight_orignal;
            assign weight_new_array[i] = weight_new;
            assign weight_grad_orignal_array[i] = weight_gradient_original;
            assign weight_grad_new_array[i] = weight_gradient_new;

            assign synarray_wdata[i*WEIGHT_WIDTH+:WEIGHT_WIDTH] = weight_new_array[i];
            assign grad_array_wdata[i*GRAD_WIDTH+:GRAD_WIDTH] = weight_grad_new_array[i];
            ffstdp_update #(
                .PRE_CNT_WIDTH (POST_NEUR_SPIKE_CNT_WIDTH),
                .POST_CNT_WIDTH(POST_NEUR_SPIKE_CNT_WIDTH),
                .WEIGHT_WIDTH  (WEIGHT_WIDTH),
                .GRAD_WIDTH    (GRAD_WIDTH),
                .GOODNESS_WIDTH(GOODNESS_WIDTH)
            ) ffstdp_update_0 (
                // Inputs
                // General
                .CLK            (CLK),
                .CTRL_TREF_EVENT(CTRL_TREF_EVENT),
                .IS_POS         (IS_POS),
                .IS_TRAIN       (IS_TRAIN),
                .AVG_GOODNESS   (AVG_GOODNESS),
                // From neuron 
                .POST_SPIKE_CNT (POST_NEUR_S_CNT_array[i]),
                .PRE_SPIKE_CNT  (PRE_NEUR_S_CNT),
                // From SRAM
                .WSYN_CURR      (weight_orignal_array[i]),
                .GRAD_CURR      (weight_grad_orignal_array[i]),
                // Output
                .WSYN_NEW       (weight_new),
                .GRAD_NEW       (weight_gradient_new)
            );
        end
    endgenerate

    // FPGA RAM IP (Single port)
    // Synaptic memory wrapper
    //     sram_synaptic u_sram_synaptic(
    //     .clka                                  (CLK                ),// input wire clka
    //     .ena                                   (CTRL_SYNARRAY_CS   ),// input 片选使能信号
    //     .wea                                   (CTRL_SYNARRAY_WE   ),// input 写使能信号
    //     .addra                                 (synarray_addr      ),
    //     .dina                                  (synarray_wdata     ),
    //     .douta                                 (SYNARRAY_RDATA     ) 
    // );

    sram_synaptic_sim #(
        .DATA_WIDTH(SYN_ARRAY_DATA_WIDTH),
        .ADDR_WIDTH(SYN_ARRAY_ADDR_WIDTH),
        .SRAM_DEPTH(INPUT_NEURON * OUTPUT_NEURON / POST_NEUR_PARALLEL)
    ) u_sram_synaptic_bank (
        // Global inputs
        .CK(CLK),               // Clock (synchronous read/write)
        // Control and data inputs
        .CS(CTRL_SYNARRAY_CS),  // Chip select
        .WE(CTRL_SYNARRAY_WE),  // Write enable
        .A (synarray_addr),     // Address bus
        .D (synarray_wdata),    // Data input bus (write)
        // Data output
        .Q (SYNARRAY_RDATA)     // Data output bus (read)
    );

    sram_synaptic_sim #(
        .DATA_WIDTH(GRAD_ARRAY_DATA_WIDTH),
        .ADDR_WIDTH(GRAD_ARRAY_ADDR_WIDTH),
        .SRAM_DEPTH(INPUT_NEURON * OUTPUT_NEURON / POST_NEUR_PARALLEL)
    ) u_sram_synaptic_gradient_bank (
        // Global inputs
        .CK(CLK),                 // Clock (synchronous read/write)
        // Control and data inputs
        .CS(CTRL_GRAD_ARRAY_CS),  // Chip select
        .WE(CTRL_GRAD_ARRAY_WE),  // Write enable
        .A (grad_array_addr),     // Address bus
        .D (grad_array_wdata),    // Data input bus (write)
        // Data output
        .Q (GRAD_ARRAY_RDATA)     // Data output bus (read)
    );

endmodule

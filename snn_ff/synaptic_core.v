module synaptic_core #(
    parameter TIME_STEP = 8,
    parameter INPUT_NEURON = 784,
    parameter OUTPUT_NEURON = 256,
    parameter AER_WIDTH = 12,

    parameter PRE_NEUR_ADDR_WIDTH = 10,
    parameter PRE_NEUR_WORD_ADDR_WIDTH= 10,
    parameter PRE_NEUR_BYTE_ADDR_WIDTH = 0,

    parameter POST_NEUR_ADDR_WIDTH = 10,
    parameter POST_NEUR_WORD_ADDR_WIDTH= 8,
    parameter POST_NEUR_BYTE_ADDR_WIDTH = 2,
    parameter POST_NEUR_PARALLEL = 4,
    
    parameter PRE_NEUR_DATA_WIDTH = 8,
    parameter POST_NEUR_DATA_WIDTH = 32,
    parameter POST_NEUR_MEM_WIDTH = 12,
    parameter POST_NEUR_SPIKE_CNT_WIDTH = 7,
    parameter SYN_ARRAY_DATA_WIDTH = 32,
    parameter SYN_ARRAY_ADDR_WIDTH = 16,
    parameter WEIGHT_WIDTH = 8
)(
    input wire IS_POS,
    input wire IS_TRAIN,
    // Global inputs ------------------------------------------
    input  wire CLK,

    // Inputs from SPI configuration registers ----------------
    input  wire SPI_GATE_ACTIVITY_sync,

    // Inputs from controller ---------------------------------
    input wire CTRL_SYNARRAY_CS,
    input wire CTRL_SYNARRAY_WE,
    input wire [SYN_ARRAY_ADDR_WIDTH-1:0] CTRL_SYNARRAY_ADDR,
    input wire [POST_NEUR_ADDR_WIDTH-1:0]  CTRL_POST_NEURON_ADDRESS,

    input wire CTRL_SYNA_WR_EVENT,
    input wire CTRL_SYNA_RD_EVENT,
    input wire[WEIGHT_WIDTH-1:0] CTRL_SYNA_PROG_DATA,
    
    input wire  CTRL_NEUR_EVENT,
    input wire  CTRL_TSTEP_EVENT,
    input wire  CTRL_TREF_EVENT,
    // Inputs from neurons ------------------------------------
    input wire [PRE_NEUR_DATA_WIDTH-1:0] PRE_NEUR_S_CNT,
    input wire [POST_NEUR_SPIKE_CNT_WIDTH * POST_NEUR_PARALLEL -1 :0] POST_NEUR_S_CNT,
    // Outputs ------------------------------------------------
    output wire [SYN_ARRAY_DATA_WIDTH-1:0] SYNARRAY_RDATA
);
    localparam SYN_SRAM_DATA_WIDTH = POST_NEUR_PARALLEL * WEIGHT_WIDTH; 
    // Internal regs and wires definitions
    wire [SYN_ARRAY_ADDR_WIDTH-1:0]    synarray_addr;
    wire [POST_NEUR_BYTE_ADDR_WIDTH-1:0]     post_neuron_byte_addr;

    wire [SYN_SRAM_DATA_WIDTH-1:0] synarray_wdata;
    wire [WEIGHT_WIDTH-1:0] weight_orignal_array[0:POST_NEUR_PARALLEL-1];
    wire [WEIGHT_WIDTH-1:0] weight_new_array[0:POST_NEUR_PARALLEL-1];

    wire [POST_NEUR_SPIKE_CNT_WIDTH-1:0]  POST_NEUR_S_CNT_array[0:POST_NEUR_PARALLEL-1];

    assign synarray_addr = CTRL_SYNARRAY_ADDR;
    assign post_neuron_byte_addr = CTRL_POST_NEURON_ADDRESS[POST_NEUR_BYTE_ADDR_WIDTH-1:0];


    genvar i;
    // SDSP update logic
    generate
        for (i=0; i<POST_NEUR_PARALLEL; i=i+1) begin: gen_ffstdp_update
        wire [WEIGHT_WIDTH-1:0] weight_orignal = SYNARRAY_RDATA[i*WEIGHT_WIDTH +: WEIGHT_WIDTH];
        wire [WEIGHT_WIDTH-1:0] weight_new;
        assign POST_NEUR_S_CNT_array[i] = POST_NEUR_S_CNT[i*POST_NEUR_SPIKE_CNT_WIDTH +: POST_NEUR_SPIKE_CNT_WIDTH];
        assign weight_orignal_array[i] = weight_orignal;
        assign weight_new_array[i] = weight_new;
        // assign synarray_wdata[(i*8)+7:(i*8)] = SPI_GATE_ACTIVITY_sync? (i==post_neuron_byte_addr && CTRL_SYNA_WR_EVENT)? CTRL_SYNA_PROG_DATA : synarray_rdata[(i*8)+7:(i*8)]
        //                                      : synarray_wdata_int[(i*8)+7:(i*8)];
        assign synarray_wdata[i*WEIGHT_WIDTH +: WEIGHT_WIDTH] = weight_new_array[i];
        ffstdp_update# 
        (
            .PRE_CNT_WIDTH       (POST_NEUR_SPIKE_CNT_WIDTH  ),
            .POST_CNT_WIDTH      (POST_NEUR_SPIKE_CNT_WIDTH  ),
            .WEIGHT_WIDTH        (WEIGHT_WIDTH               )
        )
        ffstdp_update_0(
            // Inputs
            // General
            .CLK(CLK),
            .CTRL_TREF_EVENT(CTRL_TREF_EVENT),
            .IS_POS(IS_POS),           
            .IS_TRAIN(IS_TRAIN),
            // From neuron 
            .POST_SPIKE_CNT(POST_NEUR_S_CNT_array[i]),
            .PRE_SPIKE_CNT(PRE_NEUR_S_CNT), 
            // From SRAM
            .WSYN_CURR(weight_orignal_array[i]),
            // Output
            .WSYN_NEW(weight_new)
        );
        end
    endgenerate
    
    
    // Synaptic memory wrapper
    SRAM_65536x32_wrapper SRAM_65536x32_wrapper_0(
    .clka  (CLK      ),  // input wire clka
    .ena   (CTRL_SYNARRAY_CS       ),  // input 片选使能信号
    .wea   (CTRL_SYNARRAY_WE       ),  // input 写使能信号
    .addra (synarray_addr    ), 
    .dina  (synarray_wdata  ),
    .douta (SYNARRAY_RDATA  )  
);
endmodule




// module SRAM_8192x32_wrapper (

//     // Global inputs
//     input         CK,                       // Clock (synchronous read/write)

//     // Control and data inputs
//     input         CS,                       // Chip select
//     input         WE,                       // Write enable
//     input  [12:0] A,                        // Address bus 
//     input  [31:0] D,                        // Data input bus (write)

//     // Data output
//     output [31:0] Q                         // Data output bus (read)   
// );


//     /*
//      *  Simple behavioral code for simulation, to be replaced by a 8192-word 32-bit SRAM macro 
//      *  or Block RAM (BRAM) memory with the same format for FPGA implementations.
//      */      
//         reg [31:0] SRAM[8191:0];
//         reg [31:0] Qr;
//         always @(posedge CK) begin
//             Qr <= CS ? SRAM[A] : Qr;
//             if (CS & WE) SRAM[A] <= D;
//         end
//         assign Q = Qr;
    
// endmodule

module sram_synaptic_sim #(
    parameter  ADDR_WIDTH = 8,
    parameter  DATA_WIDTH = 32,
    parameter  SRAM_DEPTH = 256
)

(

    // Global inputs
    input          CK,                       // Clock (synchronous read/write)

    // Control and data inputs
    input          CS,                       // Chip select
    input          WE,                       // Write enable
    input  [ADDR_WIDTH-1:0] A,                        // Address bus 
    input  [DATA_WIDTH-1:0] D,                        // Data input bus (write)

    // Data output
    output [DATA_WIDTH-1:0] Q                         // Data output bus (read)   
);
    /*
     *  Simple behavioral code for simulation, to be replaced by a 256-word 32-bit SRAM macro 
     *  or Block RAM (BRAM) memory with the same format for FPGA implementations.
     */      
        reg [DATA_WIDTH-1:0] SRAM[0:SRAM_DEPTH-1];
        reg [DATA_WIDTH-1:0] Qr;

        initial begin
        $readmemb("D:/OneDrive/SNN_FFSTDP/Gen_out/MNIST/weights_weight.txt", SRAM);
        end

        always @(posedge CK) begin
            Qr <= CS ? SRAM[A] : Qr;
            if (CS & WE) SRAM[A] <= D;
        end
        assign Q = Qr;


endmodule

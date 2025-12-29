`timescale 1ns / 1ps

module tb_SRAM_256x128_wrapper;

    // Parameters
    parameter ADDR_WIDTH = 8;    // SRAM address width (256 addresses)
    parameter DATA_WIDTH = 128;  // SRAM data width (128 bits per address)

    // Inputs
    reg CLK;                     // Clock signal
    reg CTRL_POST_NEUR_CS;       // Chip select signal for SRAM
    reg CTRL_POST_NEUR_WE;       // Write enable signal for SRAM
    reg [ADDR_WIDTH-1:0] post_neuron_sram_addr; // SRAM address input
    wire [DATA_WIDTH-1:0] post_neuron_sram_in;   // Data input to SRAM
    wire [DATA_WIDTH-1:0] processed_data;

    // Outputs
    wire [DATA_WIDTH-1:0] post_neuron_sram_out; // Data output from SRAM

    // Instantiate the SRAM_256x128_wrapper module
    SRAM_256x128_wrapper neurarray_post (
        .clka(CLK), 
        .ena(CTRL_POST_NEUR_CS),
        .wea(CTRL_POST_NEUR_WE),
        .addra(post_neuron_sram_addr),
        .dina(post_neuron_sram_in),
        .douta(post_neuron_sram_out)
    );

    // Clock generation
    always begin
        #5 CLK = ~CLK;  // 10 ns clock period
    end
    assign processed_data = post_neuron_sram_out + post_neuron_sram_addr;
    assign post_neuron_sram_in = processed_data;
    
    always @(posedge CLK)           
        begin                                        
            post_neuron_sram_addr <= post_neuron_sram_addr + 1;                                   
        end                                          
    // Test stimulus block
    initial begin
        // Initialize signals
        CLK = 1;
        post_neuron_sram_addr = 0;
        CTRL_POST_NEUR_CS = 0;     // Disable chip select
        CTRL_POST_NEUR_WE = 0;     // Disable write enable

        // Test 1: Write initial data to address 0
        #10;
        CTRL_POST_NEUR_CS = 1;     // Enable chip select
        CTRL_POST_NEUR_WE = 1;     // Enable write operation
        #10;
        CTRL_POST_NEUR_CS = 0;     // Enable chip select
        CTRL_POST_NEUR_WE = 0;     // Enable write operation
        #30;
        // post_neuron_sram_addr = 'd1;
        // CTRL_POST_NEUR_CS = 1;     // Enable chip select
        // CTRL_POST_NEUR_WE = 1;     // Enable write operation
        #30;
        post_neuron_sram_addr = 'd1;
        CTRL_POST_NEUR_CS = 1;     // Enable chip select
        CTRL_POST_NEUR_WE = 0;     // Enable write operation

        $stop;
    end

endmodule

`timescale 1ns / 1ps

module tb_axi4_rw_test_v1_0;

    // Parameters
    parameter TIME_TOTAL_STEP = 16;
    parameter DATA_SIZE = 784;
    parameter TOTAL_TRAIN_SIZE = 2000;
    parameter TOTAL_TEST_SIZE = 1000;
    parameter C_M_TARGET_SLAVE_WRITE_ADDR = 32'h01900000;
    parameter C_M_TARGET_SLAVE_READ_ADDR = 32'h00010000;
    parameter integer C_M_AXI_READ_BURST_LEN = 4;
    parameter integer C_M_AXI_WRITE_BURST_LEN = 1;
    parameter integer C_M_AXI_ID_WIDTH = 1;
    parameter integer C_M_AXI_ADDR_WIDTH = 32;
    parameter integer C_M_AXI_DATA_WIDTH = 32;
    parameter integer C_M_AXI_AWUSER_WIDTH = 0;
    parameter integer C_M_AXI_ARUSER_WIDTH = 0;
    parameter integer C_M_AXI_WUSER_WIDTH = 0;
    parameter integer C_M_AXI_RUSER_WIDTH = 0;
    parameter integer C_M_AXI_BUSER_WIDTH = 0;

    // Inputs (reg 类型)
    wire PROCESS_DONE;
    wire [31:0] GOODNESS;
    reg SNN_CLK;
    wire AER_IN_ACK;
    reg m_axi_init_axi_txn;
    reg m_axi_aclk;
    reg m_axi_aresetn;
    reg m_axi_awready;
    reg m_axi_wready;
    reg [C_M_AXI_ID_WIDTH-1:0] m_axi_bid;
    reg [1:0] m_axi_bresp;
    reg [C_M_AXI_BUSER_WIDTH-1:0] m_axi_buser;
    reg m_axi_bvalid;
    reg m_axi_arready;
    reg [C_M_AXI_ID_WIDTH-1:0] m_axi_rid;
    reg [C_M_AXI_DATA_WIDTH-1:0] m_axi_rdata;
    reg [1:0] m_axi_rresp;
    reg m_axi_rlast;
    reg [C_M_AXI_RUSER_WIDTH-1:0] m_axi_ruser;
    reg m_axi_rvalid;

    logic auto_ack_verbose;
    logic [7:0] aer_neur_spk;

    // Outputs (wire 类型)
    logic AER_IN_REQ;
    logic [11:0] AER_IN_ADDR;
    wire IS_POS;
    wire IS_TRAIN;
    wire [2:0] STATE;
    wire m_axi_txn_done;
    wire m_axi_error;
    wire [C_M_AXI_ID_WIDTH-1:0] m_axi_awid;
    wire [C_M_AXI_ADDR_WIDTH-1:0] m_axi_awaddr;
    wire [7:0] m_axi_awlen;
    wire [2:0] m_axi_awsize;
    wire [1:0] m_axi_awburst;
    wire m_axi_awlock;
    wire [3:0] m_axi_awcache;
    wire [2:0] m_axi_awprot;
    wire [3:0] m_axi_awqos;
    wire [C_M_AXI_AWUSER_WIDTH-1:0] m_axi_awuser;
    wire m_axi_awvalid;
    wire [C_M_AXI_DATA_WIDTH-1:0] m_axi_wdata;
    wire [C_M_AXI_DATA_WIDTH/8-1:0] m_axi_wstrb;
    wire m_axi_wlast;
    wire [C_M_AXI_WUSER_WIDTH-1:0] m_axi_wuser;
    wire m_axi_wvalid;
    wire m_axi_bready;
    wire [C_M_AXI_ID_WIDTH-1:0] m_axi_arid;
    wire [C_M_AXI_ADDR_WIDTH-1:0] m_axi_araddr;
    wire [7:0] m_axi_arlen;
    wire [2:0] m_axi_arsize;
    wire [1:0] m_axi_arburst;
    wire m_axi_arlock;
    wire [3:0] m_axi_arcache;
    wire [2:0] m_axi_arprot;
    wire [3:0] m_axi_arqos;
    wire [C_M_AXI_ARUSER_WIDTH-1:0] m_axi_aruser;
    wire m_axi_arvalid;
    wire m_axi_rready;
    reg m_axi_rready_reg;
    reg[31:0] cnt;


axi4_rw_test_v1_0#(
   .TIME_TOTAL_STEP(16             ),
   .DATA_SIZE      (784            ),
   .TOTAL_TRAIN_SIZE(2000             ),
   .TOTAL_TEST_SIZE(1000           ),
   .C_M_TARGET_SLAVE_WRITE_ADDR(32'h01900000   ),
   .C_M_TARGET_SLAVE_READ_ADDR(32'h00010000   )
)
 u_axi4_rw_test_v1_0(
// Users to add parameters here
// User parameters ends
// Do not modify the parameters beyond this line
// Parameters of Axi Master Bus Interface M_AXI
// Users to add ports here
    .PROCESS_DONE                       (PROCESS_DONE              ),
    .GOODNESS                           (GOODNESS                  ),
    .SNN_CLK                            (SNN_CLK                   ),
    .AER_IN_ACK                         (AER_IN_ACK                ),
    .AER_IN_REQ                         (AER_IN_REQ                ),
    .AER_IN_ADDR                        (AER_IN_ADDR               ),
    .IS_POS                             (IS_POS                    ),
    .IS_TRAIN                           (IS_TRAIN                  ),
// Debug
    .STATE                              (STATE                     ),
// User ports ends
// Do not modify the ports beyond this line
// Ports of Axi Master Bus Interface M_AXI
    .m_axi_init_axi_txn                 (m_axi_init_axi_txn        ),
    .TRAIN_ENABLE_FLAG                  (m_axi_txn_done            ),
    .TRAIN_FINISH_FLAG                  (m_axi_error               ),
    .m_axi_aclk                         (m_axi_aclk                ),
    .m_axi_aresetn                      (m_axi_aresetn             ),
    .m_axi_awid                         (m_axi_awid                ),
    .m_axi_awaddr                       (m_axi_awaddr              ),
    .m_axi_awlen                        (m_axi_awlen               ),
    .m_axi_awsize                       (m_axi_awsize              ),
    .m_axi_awburst                      (m_axi_awburst             ),
    .m_axi_awlock                       (m_axi_awlock              ),
    .m_axi_awcache                      (m_axi_awcache             ),
    .m_axi_awprot                       (m_axi_awprot              ),
    .m_axi_awqos                        (m_axi_awqos               ),
    .m_axi_awuser                       (m_axi_awuser              ),
    .m_axi_awvalid                      (m_axi_awvalid             ),
    .m_axi_awready                      (m_axi_awready             ),
    .m_axi_wdata                        (m_axi_wdata               ),
    .m_axi_wstrb                        (m_axi_wstrb               ),
    .m_axi_wlast                        (m_axi_wlast               ),
    .m_axi_wuser                        (m_axi_wuser               ),
    .m_axi_wvalid                       (m_axi_wvalid              ),
    .m_axi_wready                       (m_axi_wready              ),
    .m_axi_bid                          (m_axi_bid                 ),
    .m_axi_bresp                        (m_axi_bresp               ),
    .m_axi_buser                        (m_axi_buser               ),
    .m_axi_bvalid                       (m_axi_bvalid              ),
    .m_axi_bready                       (m_axi_bready              ),
    .m_axi_arid                         (m_axi_arid                ),
    .m_axi_araddr                       (m_axi_araddr              ),
    .m_axi_arlen                        (m_axi_arlen               ),
    .m_axi_arsize                       (m_axi_arsize              ),
    .m_axi_arburst                      (m_axi_arburst             ),
    .m_axi_arlock                       (m_axi_arlock              ),
    .m_axi_arcache                      (m_axi_arcache             ),
    .m_axi_arprot                       (m_axi_arprot              ),
    .m_axi_arqos                        (m_axi_arqos               ),
    .m_axi_aruser                       (m_axi_aruser              ),
    .m_axi_arvalid                      (m_axi_arvalid             ),
    .m_axi_arready                      (m_axi_arready             ),
    .m_axi_rid                          (m_axi_rid                 ),
    .m_axi_rdata                        (m_axi_rdata               ),
    .m_axi_rresp                        (m_axi_rresp               ),
    .m_axi_rlast                        (m_axi_rlast               ),
    .m_axi_ruser                        (m_axi_ruser               ),
    .m_axi_rvalid                       (m_axi_rvalid              ),
    .m_axi_rready                       (m_axi_rready              )
);

Top_test u_Top_test(
    .CLK                                (SNN_CLK                       ),
    .RST                                (!m_axi_aresetn                   ),
    .AERIN_ADDR                         (AER_IN_ADDR                ),
    .AERIN_REQ                          (AER_IN_REQ                 ),
    .IS_POS                             (IS_POS                    ),
    .IS_TRAIN                           (1'b1                  ),
    .AERIN_ACK                          (AER_IN_ACK                 ),
    .GOODNESS                           (GOODNESS                  ),
    .PROCESS_DONE                       (PROCESS_DONE              )
);


    // 时钟产生
    always #5 m_axi_aclk = ~m_axi_aclk;
    
    always @(posedge m_axi_aclk) begin
      m_axi_rready_reg <= m_axi_rready;
    end

parameter int N = 3000;   // 样本�?
parameter int T = 16;   // 时间�?
parameter int WIDTH = 784;  // 每个时间步的 bit �?
parameter int RAM_DEPTH = (N * T * WIDTH) / 32;  // 计算 RAM 深度，每 32 位为�?�?
integer file, byte_count;
bit [7:0] spike_bytes [0:3]; // 存储 4 个字�?
bit [7:0]  ram [0:RAM_DEPTH-1]; // 定义 RAM 存储�?
bit [31:0] rd_burst_data [0:3];
integer bit_index = 0;
integer ram_index = 0;
integer i, block_index = 0;

    initial begin
        // 读取 HEX 文件�? RAM
        $readmemh("D:/OneDrive/SNN_FFSTBP/sim/python/all_spikes.hex", ram);

        // 打印部分数据用于�?�?
        for (i = 0; i < 16; i = i + 1) begin
            $display("RAM[%0d] = %h", i, ram[i]);
        end

        $display("Spike data loaded successfully!");
    end

  
    // always@(posedge m_axi_aclk) begin
    //     if (cnt == 'd784)
    //         cnt <= 0;
    //     else if(AER_IN_REQ && AER_IN_ACK)
    //         cnt <= cnt + 1;
    //     else
    //         cnt <= cnt;
    // end

    // 声明数据存储
    // logic [C_M_AXI_DATA_WIDTH-1:0] write_data[4] = '{32'hDEADBEEF, 32'h12345678, 32'h87654321, 32'hCAFEBABE};
    // logic [C_M_AXI_DATA_WIDTH-1:0] read_data[4];

    // // 写入 AXI（地�? 0x1000，突发长�? 4�?
    // axi_write_task(32'h00001000, write_data, 4);

    // // 读取 AXI（地�? 0x1000，突发长�? 4�?
    // axi_read_task(32'h00001000, 4, read_data);
    initial begin
    //   SNN_CLK = 0;
    //   #2;
      SNN_CLK = 1;
      forever
        #5 SNN_CLK = ~SNN_CLK;
    end
    always @(posedge m_axi_aclk) begin
        if(m_axi_wvalid) begin
            m_axi_bvalid <= 1;
            m_axi_bresp  <= 2'b00;
        end
        else if(m_axi_bready)begin
            m_axi_bvalid <= 0;
        end
        else begin
            m_axi_bvalid <= m_axi_bvalid;
        end
    end
    // 测试过程
    initial begin
        // 初始化信�?
        m_axi_aclk = 1;
        m_axi_init_axi_txn = 0;
        m_axi_aresetn = 0;
        m_axi_awready = 1;
        m_axi_wready = 1;
        m_axi_bid = 0;
        m_axi_bresp = 0;
        m_axi_buser = 0;
        m_axi_bvalid = 0;
        m_axi_arready = 1;
        m_axi_rid = 0;
        m_axi_rdata = 0;
        m_axi_rresp = 0;
        m_axi_rlast = 0;
        m_axi_ruser = 0;
        m_axi_rvalid = 0;

        // 复位
        #10 m_axi_aresetn = 1;
        auto_ack_verbose = 1'b1;
        // fork
        //   auto_ack(.req(AER_IN_REQ), .ack(AER_IN_ACK), .addr(AER_IN_ADDR), .neur(aer_neur_spk), .verbose(auto_ack_verbose));
        // join_none
        // 生成写事�?

        // 生成读事�?
        // fork
        //     axi_slave_read_response();
        // join_none

        #200 m_axi_init_axi_txn = 1;
        #10 m_axi_init_axi_txn = 0;

        // 进行AXI读写操作测试
        // #50 m_axi_awready = 1;
        // #50 m_axi_wready = 1;
        // #50 m_axi_bvalid = 1;
        // #50 m_axi_arready = 1;
        // #50 m_axi_rvalid = 1; m_axi_rdata = 32'h12345678;
        
        wait_ns(10);
        while(STATE != 3'b000) begin
          while(STATE != 3'b011) begin
                for (i = 0; i < 4; i = i + 1) begin
                // 按大端模式拼�? 32 位数�?
                rd_burst_data[i] = {ram[block_index * 16 + i * 4+3], 
                               ram[block_index * 16 + i * 4 + 2], 
                               ram[block_index * 16 + i * 4 + 1], 
                               ram[block_index * 16 + i * 4]};
                end
                block_index++;
                axi_slave_read_response(.ram_data(rd_burst_data));
                @(posedge m_axi_aclk);
           end
        //    wait (m_axi_awvalid);
        //         m_axi_bvalid <= 1;
        //    wait (m_axi_bready);
        //         m_axi_bvalid <= 0;
            // wait(PROCESS_DONE);
            // axi_slave_write_response();
        end


        #100 $stop;
    end
// ---------------- AXI 从机（Slave�? 写响�? TASK ----------------
// task axi_slave_write_response();
//     int i;
    
//     // 1. 等待写地�?有效
//     wait (m_axi_awvalid);
//     @(posedge m_axi_aclk) m_axi_awready <= 1; // 发�?�写地址握手信号
//     @(posedge m_axi_aclk) m_axi_awready <= 0; // 释放握手信号

//     // 2. 等待写数�?
//     for (i = 0; i < m_axi_awlen + 1; i++) begin  // `awlen` �? 0-based 计数
//         wait (m_axi_wvalid);
//         @(posedge m_axi_aclk) m_axi_wready <= 1; // 允许写数�?
//         @(posedge m_axi_aclk) m_axi_wready <= 0;
//     end

//     // 3. 发�?�写响应
//     @(posedge m_axi_aclk) 
//     begin
//         m_axi_bvalid <= 1;
//         m_axi_bresp  <= 2'b00; // OKAY
//     end
//     // 等待主机 `bready` 信号
//     wait (m_axi_bready);
//     @(posedge m_axi_aclk) m_axi_bvalid <= 0; // 清除 `bvalid`
// endtask


// ---------------- AXI 从机（Slave�? 读响�? TASK ----------------
task axi_slave_read_response(
    input bit [31:0] ram_data [0:3]
);
    int i;
    
    // 1. 等待主机发�?�读地址
    wait (m_axi_arvalid);
    @(posedge m_axi_aclk);
    #1;
    // m_axi_arready = 1;
    // @(posedge m_axi_aclk);
    // m_axi_arready = 0;

    // 2. 发�?�读数据
    for (i = 0; i <= m_axi_arlen; i++) begin
        // @(posedge m_axi_aclk);
        m_axi_rvalid = 1;
        m_axi_rdata  = ram_data[i]; // 示例数据
        m_axi_rlast  = (i == m_axi_arlen) ? 1 : 0;
        wait (m_axi_rready);
        @(posedge m_axi_aclk);
        #1;
        m_axi_rvalid = 0;
        m_axi_rlast = 0;
    end
  endtask

task automatic auto_ack (
      ref    logic       req,
      ref    logic       ack,
      ref    logic [11:0] addr,
      ref    logic [11:0] neur,
      ref    logic       verbose
  );
  
      forever begin
          while (~req) wait_ns(1);
          wait_ns(100);
          neur = addr;
          if (verbose)
              $display("----- NEURON OUTPUT SPIKE (FROM AER): Event from neuron %d", neur);
          ack = 1'b1;
          while (req) wait_ns(1);
          wait_ns(100);
          ack = 1'b0;
      end
endtask
task wait_ns;
  input   tics_ns;
  integer tics_ns;
  #tics_ns;
endtask
endmodule
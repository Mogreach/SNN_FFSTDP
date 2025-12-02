
`timescale 1 ns / 1 ps

	module axi4_rw_test_v1_0 #
	(
		// Users to add parameters here
		parameter TIME_TOTAL_STEP = 16,
		parameter DATA_SIZE = 784,
		parameter TOTAL_TRAIN_SIZE = 2 * 1000, // 训练样本数（含正负样本）
		parameter TOTAL_TEST_SIZE  = 10* 100, //  测试样本数
		parameter  C_M_TARGET_SLAVE_READ_ADDR	= 32'h00200000, // 读基地址
		parameter  C_M_TARGET_SLAVE_WRITE_ADDR  = 32'h01900000, // 写基地址
		parameter integer C_M_AXI_READ_BURST_LEN	= 4, // 读突发长度
		parameter integer C_M_AXI_WRITE_BURST_LEN	= 1, // 写突发长度
		// User parameters ends
		// Do not modify the parameters beyond this line
		// Parameters of Axi Master Bus Interface M_AXI
		parameter integer C_M_AXI_ID_WIDTH	= 1,
		parameter integer C_M_AXI_ADDR_WIDTH	= 32,
		parameter integer C_M_AXI_DATA_WIDTH	= 32,
		parameter integer C_M_AXI_AWUSER_WIDTH	= 0,
		parameter integer C_M_AXI_ARUSER_WIDTH	= 0,
		parameter integer C_M_AXI_WUSER_WIDTH	= 0,
		parameter integer C_M_AXI_RUSER_WIDTH	= 0,
		parameter integer C_M_AXI_BUSER_WIDTH	= 0
	)
	(
		// Users to add ports here
		input  wire                         PROCESS_DONE               ,
		input  wire        [  31: 0]        GOODNESS                   ,
		input  wire                         SNN_CLK                    ,
		input  wire                         AER_IN_ACK                 ,
		output wire                         AER_IN_REQ                 ,
		output wire        [  11: 0]        AER_IN_ADDR                ,
    	output wire                         IS_POS                     ,
		output wire   						IS_TRAIN                   ,
		input wire  m_axi_init_axi_txn,
		input wire  train_enable_txn,
		// Debug
		output wire        [   2: 0]        STATE                      ,
		
		// User ports ends
		// Do not modify the ports beyond this line


		// Ports of Axi Master Bus Interface M_AXI

		output wire  TRAIN_ENABLE_FLAG,
		output wire  TRAIN_FINISH_FLAG,
		input wire  m_axi_aclk,
		input wire  m_axi_aresetn,
		output wire [C_M_AXI_ID_WIDTH-1 : 0] m_axi_awid,
		output wire [C_M_AXI_ADDR_WIDTH-1 : 0] m_axi_awaddr,
		output wire [7 : 0] m_axi_awlen,
		output wire [2 : 0] m_axi_awsize,
		output wire [1 : 0] m_axi_awburst,
		output wire  m_axi_awlock,
		output wire [3 : 0] m_axi_awcache,
		output wire [2 : 0] m_axi_awprot,
		output wire [3 : 0] m_axi_awqos,
		output wire [C_M_AXI_AWUSER_WIDTH-1 : 0] m_axi_awuser,
		output wire  m_axi_awvalid,
		input wire  m_axi_awready,
		output wire [C_M_AXI_DATA_WIDTH-1 : 0] m_axi_wdata,
		output wire [C_M_AXI_DATA_WIDTH/8-1 : 0] m_axi_wstrb,
		output wire  m_axi_wlast,
		output wire [C_M_AXI_WUSER_WIDTH-1 : 0] m_axi_wuser,
		output wire  m_axi_wvalid,
		input wire  m_axi_wready,
		input wire [C_M_AXI_ID_WIDTH-1 : 0] m_axi_bid,
		input wire [1 : 0] m_axi_bresp,
		input wire [C_M_AXI_BUSER_WIDTH-1 : 0] m_axi_buser,
		input wire  m_axi_bvalid,
		output wire  m_axi_bready,
		output wire [C_M_AXI_ID_WIDTH-1 : 0] m_axi_arid,
		output wire [C_M_AXI_ADDR_WIDTH-1 : 0] m_axi_araddr,
		output wire [7 : 0] m_axi_arlen,
		output wire [2 : 0] m_axi_arsize,
		output wire [1 : 0] m_axi_arburst,
		output wire  m_axi_arlock,
		output wire [3 : 0] m_axi_arcache,
		output wire [2 : 0] m_axi_arprot,
		output wire [3 : 0] m_axi_arqos,
		output wire [C_M_AXI_ARUSER_WIDTH-1 : 0] m_axi_aruser,
		output wire  m_axi_arvalid,
		input wire  m_axi_arready,
		input wire [C_M_AXI_ID_WIDTH-1 : 0] m_axi_rid,
		input wire [C_M_AXI_DATA_WIDTH-1 : 0] m_axi_rdata,
		input wire [1 : 0] m_axi_rresp,
		input wire  m_axi_rlast,
		input wire [C_M_AXI_RUSER_WIDTH-1 : 0] m_axi_ruser,
		input wire  m_axi_rvalid,
		output wire  m_axi_rready
	);
// Instantiation of Axi Bus Interface M_AXI
	axi4_rw_test_v1_0_M_AXI # ( 
		.TIME_TOTAL_STEP(TIME_TOTAL_STEP),
		.DATA_SIZE      (DATA_SIZE),
		.TOTAL_TRAIN_SIZE(TOTAL_TRAIN_SIZE),
		.TOTAL_TEST_SIZE(TOTAL_TEST_SIZE),
		.C_M_TARGET_SLAVE_WRITE_ADDR(C_M_TARGET_SLAVE_WRITE_ADDR),
		.C_M_TARGET_SLAVE_READ_ADDR(C_M_TARGET_SLAVE_READ_ADDR),
		.C_M_AXI_READ_BURST_LEN(C_M_AXI_READ_BURST_LEN),
		.C_M_AXI_WRITE_BURST_LEN(C_M_AXI_WRITE_BURST_LEN),
		
		.C_M_AXI_ID_WIDTH(C_M_AXI_ID_WIDTH),
		.C_M_AXI_ADDR_WIDTH(C_M_AXI_ADDR_WIDTH),
		.C_M_AXI_DATA_WIDTH(C_M_AXI_DATA_WIDTH),
		.C_M_AXI_AWUSER_WIDTH(C_M_AXI_AWUSER_WIDTH),
		.C_M_AXI_ARUSER_WIDTH(C_M_AXI_ARUSER_WIDTH),
		.C_M_AXI_WUSER_WIDTH(C_M_AXI_WUSER_WIDTH),
		.C_M_AXI_RUSER_WIDTH(C_M_AXI_RUSER_WIDTH),
		.C_M_AXI_BUSER_WIDTH(C_M_AXI_BUSER_WIDTH)
	) axi4_rw_test_v1_0_M_AXI_inst (
		// input from SNN Module
		.PROCESS_DONE                       (PROCESS_DONE              ),
		.GOODNESS                           (GOODNESS                  ),
		.SNN_CLK                            (SNN_CLK                   ),
		.AER_IN_ACK                         (AER_IN_ACK                ),
		.AER_IN_REQ                         (AER_IN_REQ                ),
		.AER_IN_ADDR                        (AER_IN_ADDR               ),
		.IS_POS                             (IS_POS                    ),
		.IS_TRAIN                           (IS_TRAIN                  ),
		.train_enable_txn                   (train_enable_txn          ),
		// Debug
    	.STATE                              (STATE                     ),

		.INIT_AXI_TXN(m_axi_init_axi_txn),
		.TRAIN_ENABLE_FLAG(TRAIN_ENABLE_FLAG),
		.TRAIN_FINISH_FLAG(TRAIN_FINISH_FLAG),
		.M_AXI_ACLK(m_axi_aclk),
		.M_AXI_ARESETN(m_axi_aresetn),
		.M_AXI_AWID(m_axi_awid),
		.M_AXI_AWADDR(m_axi_awaddr),
		.M_AXI_AWLEN(m_axi_awlen),
		.M_AXI_AWSIZE(m_axi_awsize),
		.M_AXI_AWBURST(m_axi_awburst),
		.M_AXI_AWLOCK(m_axi_awlock),
		.M_AXI_AWCACHE(m_axi_awcache),
		.M_AXI_AWPROT(m_axi_awprot),
		.M_AXI_AWQOS(m_axi_awqos),
		.M_AXI_AWUSER(m_axi_awuser),
		.M_AXI_AWVALID(m_axi_awvalid),
		.M_AXI_AWREADY(m_axi_awready),
		.M_AXI_WDATA(m_axi_wdata),
		.M_AXI_WSTRB(m_axi_wstrb),
		.M_AXI_WLAST(m_axi_wlast),
		.M_AXI_WUSER(m_axi_wuser),
		.M_AXI_WVALID(m_axi_wvalid),
		.M_AXI_WREADY(m_axi_wready),
		.M_AXI_BID(m_axi_bid),
		.M_AXI_BRESP(m_axi_bresp),
		.M_AXI_BUSER(m_axi_buser),
		.M_AXI_BVALID(m_axi_bvalid),
		.M_AXI_BREADY(m_axi_bready),
		.M_AXI_ARID(m_axi_arid),
		.M_AXI_ARADDR(m_axi_araddr),
		.M_AXI_ARLEN(m_axi_arlen),
		.M_AXI_ARSIZE(m_axi_arsize),
		.M_AXI_ARBURST(m_axi_arburst),
		.M_AXI_ARLOCK(m_axi_arlock),
		.M_AXI_ARCACHE(m_axi_arcache),
		.M_AXI_ARPROT(m_axi_arprot),
		.M_AXI_ARQOS(m_axi_arqos),
		.M_AXI_ARUSER(m_axi_aruser),
		.M_AXI_ARVALID(m_axi_arvalid),
		.M_AXI_ARREADY(m_axi_arready),
		.M_AXI_RID(m_axi_rid),
		.M_AXI_RDATA(m_axi_rdata),
		.M_AXI_RRESP(m_axi_rresp),
		.M_AXI_RLAST(m_axi_rlast),
		.M_AXI_RUSER(m_axi_ruser),
		.M_AXI_RVALID(m_axi_rvalid),
		.M_AXI_RREADY(m_axi_rready)
	);

	// Add user logic here

	// User logic ends

	endmodule
	

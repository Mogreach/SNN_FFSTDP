`ifndef TOP_TEST_CFG_VH
`define TOP_TEST_CFG_VH

// Compile-time upper bound for layer count in the pure-Verilog wrapper.
// Increase this value if you need deeper networks.
`ifndef TOP_TEST_MAX_LAYERS
`define TOP_TEST_MAX_LAYERS 8
`endif

// Bit width of each layer config item.
// 16 bits is enough for neuron counts / parallel factors in the current design.
`ifndef TOP_TEST_CFG_ITEM_WIDTH
`define TOP_TEST_CFG_ITEM_WIDTH 16
`endif

`define TOP_TEST_CFG_WIDTH (`TOP_TEST_MAX_LAYERS * `TOP_TEST_CFG_ITEM_WIDTH)

// Helper macros:
// - Item 0 corresponds to the first instantiated layer.
// - Please pass width-qualified literals, e.g. 16'd256, 16'd8.
// - These macros are valid when the number of provided items <= TOP_TEST_MAX_LAYERS.
`define TOP_TEST_CFG1(v0)                           {{(`TOP_TEST_CFG_WIDTH-(1*`TOP_TEST_CFG_ITEM_WIDTH)){1'b0}}, v0}
`define TOP_TEST_CFG2(v0,v1)                        {{(`TOP_TEST_CFG_WIDTH-(2*`TOP_TEST_CFG_ITEM_WIDTH)){1'b0}}, v1, v0}
`define TOP_TEST_CFG3(v0,v1,v2)                     {{(`TOP_TEST_CFG_WIDTH-(3*`TOP_TEST_CFG_ITEM_WIDTH)){1'b0}}, v2, v1, v0}
`define TOP_TEST_CFG4(v0,v1,v2,v3)                  {{(`TOP_TEST_CFG_WIDTH-(4*`TOP_TEST_CFG_ITEM_WIDTH)){1'b0}}, v3, v2, v1, v0}
`define TOP_TEST_CFG5(v0,v1,v2,v3,v4)               {{(`TOP_TEST_CFG_WIDTH-(5*`TOP_TEST_CFG_ITEM_WIDTH)){1'b0}}, v4, v3, v2, v1, v0}
`define TOP_TEST_CFG6(v0,v1,v2,v3,v4,v5)            {{(`TOP_TEST_CFG_WIDTH-(6*`TOP_TEST_CFG_ITEM_WIDTH)){1'b0}}, v5, v4, v3, v2, v1, v0}
`define TOP_TEST_CFG7(v0,v1,v2,v3,v4,v5,v6)         {{(`TOP_TEST_CFG_WIDTH-(7*`TOP_TEST_CFG_ITEM_WIDTH)){1'b0}}, v6, v5, v4, v3, v2, v1, v0}
`define TOP_TEST_CFG8(v0,v1,v2,v3,v4,v5,v6,v7)      {v7, v6, v5, v4, v3, v2, v1, v0}

`endif

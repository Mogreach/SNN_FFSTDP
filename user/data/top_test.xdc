create_clock -period 10.000 -name CLK_100M -waveform {0.000 5.000} [get_ports -filter { NAME =~  "*CLK*" && DIRECTION == "IN" }]

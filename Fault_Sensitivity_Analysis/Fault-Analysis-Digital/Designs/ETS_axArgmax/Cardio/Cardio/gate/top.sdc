###################################################################

# Created by write_sdc on Thu Dec 14 00:46:45 2023

###################################################################
set sdc_version 2.1

set_units -time ns -resistance kOhm -capacitance pF -voltage V -current mA
set_max_area 0
create_clock -name clk  -period 2.0e+08  -waveform {0 1.0e+08}
set_input_delay -clock clk  0  [get_ports {inp[83]}]
set_input_delay -clock clk  0  [get_ports {inp[82]}]
set_input_delay -clock clk  0  [get_ports {inp[81]}]
set_input_delay -clock clk  0  [get_ports {inp[80]}]
set_input_delay -clock clk  0  [get_ports {inp[79]}]
set_input_delay -clock clk  0  [get_ports {inp[78]}]
set_input_delay -clock clk  0  [get_ports {inp[77]}]
set_input_delay -clock clk  0  [get_ports {inp[76]}]
set_input_delay -clock clk  0  [get_ports {inp[75]}]
set_input_delay -clock clk  0  [get_ports {inp[74]}]
set_input_delay -clock clk  0  [get_ports {inp[73]}]
set_input_delay -clock clk  0  [get_ports {inp[72]}]
set_input_delay -clock clk  0  [get_ports {inp[71]}]
set_input_delay -clock clk  0  [get_ports {inp[70]}]
set_input_delay -clock clk  0  [get_ports {inp[69]}]
set_input_delay -clock clk  0  [get_ports {inp[68]}]
set_input_delay -clock clk  0  [get_ports {inp[67]}]
set_input_delay -clock clk  0  [get_ports {inp[66]}]
set_input_delay -clock clk  0  [get_ports {inp[65]}]
set_input_delay -clock clk  0  [get_ports {inp[64]}]
set_input_delay -clock clk  0  [get_ports {inp[63]}]
set_input_delay -clock clk  0  [get_ports {inp[62]}]
set_input_delay -clock clk  0  [get_ports {inp[61]}]
set_input_delay -clock clk  0  [get_ports {inp[60]}]
set_input_delay -clock clk  0  [get_ports {inp[59]}]
set_input_delay -clock clk  0  [get_ports {inp[58]}]
set_input_delay -clock clk  0  [get_ports {inp[57]}]
set_input_delay -clock clk  0  [get_ports {inp[56]}]
set_input_delay -clock clk  0  [get_ports {inp[55]}]
set_input_delay -clock clk  0  [get_ports {inp[54]}]
set_input_delay -clock clk  0  [get_ports {inp[53]}]
set_input_delay -clock clk  0  [get_ports {inp[52]}]
set_input_delay -clock clk  0  [get_ports {inp[51]}]
set_input_delay -clock clk  0  [get_ports {inp[50]}]
set_input_delay -clock clk  0  [get_ports {inp[49]}]
set_input_delay -clock clk  0  [get_ports {inp[48]}]
set_input_delay -clock clk  0  [get_ports {inp[47]}]
set_input_delay -clock clk  0  [get_ports {inp[46]}]
set_input_delay -clock clk  0  [get_ports {inp[45]}]
set_input_delay -clock clk  0  [get_ports {inp[44]}]
set_input_delay -clock clk  0  [get_ports {inp[43]}]
set_input_delay -clock clk  0  [get_ports {inp[42]}]
set_input_delay -clock clk  0  [get_ports {inp[41]}]
set_input_delay -clock clk  0  [get_ports {inp[40]}]
set_input_delay -clock clk  0  [get_ports {inp[39]}]
set_input_delay -clock clk  0  [get_ports {inp[38]}]
set_input_delay -clock clk  0  [get_ports {inp[37]}]
set_input_delay -clock clk  0  [get_ports {inp[36]}]
set_input_delay -clock clk  0  [get_ports {inp[35]}]
set_input_delay -clock clk  0  [get_ports {inp[34]}]
set_input_delay -clock clk  0  [get_ports {inp[33]}]
set_input_delay -clock clk  0  [get_ports {inp[32]}]
set_input_delay -clock clk  0  [get_ports {inp[31]}]
set_input_delay -clock clk  0  [get_ports {inp[30]}]
set_input_delay -clock clk  0  [get_ports {inp[29]}]
set_input_delay -clock clk  0  [get_ports {inp[28]}]
set_input_delay -clock clk  0  [get_ports {inp[27]}]
set_input_delay -clock clk  0  [get_ports {inp[26]}]
set_input_delay -clock clk  0  [get_ports {inp[25]}]
set_input_delay -clock clk  0  [get_ports {inp[24]}]
set_input_delay -clock clk  0  [get_ports {inp[23]}]
set_input_delay -clock clk  0  [get_ports {inp[22]}]
set_input_delay -clock clk  0  [get_ports {inp[21]}]
set_input_delay -clock clk  0  [get_ports {inp[20]}]
set_input_delay -clock clk  0  [get_ports {inp[19]}]
set_input_delay -clock clk  0  [get_ports {inp[18]}]
set_input_delay -clock clk  0  [get_ports {inp[17]}]
set_input_delay -clock clk  0  [get_ports {inp[16]}]
set_input_delay -clock clk  0  [get_ports {inp[15]}]
set_input_delay -clock clk  0  [get_ports {inp[14]}]
set_input_delay -clock clk  0  [get_ports {inp[13]}]
set_input_delay -clock clk  0  [get_ports {inp[12]}]
set_input_delay -clock clk  0  [get_ports {inp[11]}]
set_input_delay -clock clk  0  [get_ports {inp[10]}]
set_input_delay -clock clk  0  [get_ports {inp[9]}]
set_input_delay -clock clk  0  [get_ports {inp[8]}]
set_input_delay -clock clk  0  [get_ports {inp[7]}]
set_input_delay -clock clk  0  [get_ports {inp[6]}]
set_input_delay -clock clk  0  [get_ports {inp[5]}]
set_input_delay -clock clk  0  [get_ports {inp[4]}]
set_input_delay -clock clk  0  [get_ports {inp[3]}]
set_input_delay -clock clk  0  [get_ports {inp[2]}]
set_input_delay -clock clk  0  [get_ports {inp[1]}]
set_input_delay -clock clk  0  [get_ports {inp[0]}]
set_output_delay -clock clk  0  [get_ports {out[1]}]
set_output_delay -clock clk  0  [get_ports {out[0]}]

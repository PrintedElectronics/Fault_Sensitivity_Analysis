/////////////////////////////////////////////////////////////
// Created by: Synopsys DC Ultra(TM) in wire load mode
// Version   : S-2021.06-SP2
// Date      : Thu Dec  7 21:33:03 2023
/////////////////////////////////////////////////////////////


module top ( inp, out );
  input [15:0] inp;
  output [1:0] out;
  wire   n64, n65, n66, n67, n68, n69, n70, n71, n72, n73, n74, n75, n76, n77,
         n78, n79, n80, n81, n82, n83, n84, n85, n86, n87, n88, n89, n90, n91,
         n92, n93, n94, n95, n96, n97, n98, n99, n100, n101, n102, n103, n104,
         n105, n106, n107, n108, n109, n110, n111, n112, n113, n114, n115,
         n116, n117, n118, n119, n120, n121, n122, n123, n124, n125, n126,
         n127, n128, n129, n130, n131, n132, n133, n134, n135, n136, n137,
         n138, n139, n140, n141, n142, n143, n144, n145;

  NAND2X1 U106 ( .A1(n99), .A2(inp[10]), .Y(n64) );
  NAND2X1 U107 ( .A1(n109), .A2(inp[15]), .Y() );
  NOR2X1 U108 ( .A1(n137), .A2(n75), .Y(n66) );
  INVX1 U109 ( .A(n71), .Y(n67) );
  NAND2X1 U110 ( .A1(n89), .A2(n67), .Y(n68) );
  NAND2X1 U111 ( .A1(n98), .A2(inp[14]), .Y(n69) );
  AND2X1 U112 ( .A1(n69), .A2(n120), .Y(n76) );
  INVX1 U113 ( .A(n76), .Y(n70) );
  OR2X1 U114 ( .A1(inp[5]), .A2(n102), .Y(n71) );
  NAND2X1 U115 ( .A1(n102), .A2(inp[5]), .Y(n72) );
  NAND2X1 U116 ( .A1(n114), .A2(n94), .Y(n73) );
  NAND2X1 U117 ( .A1(1'b1), .A2(n84), .Y(n74) );
  NAND2X1 U118 ( .A1(n136), .A2(1'b0), .Y(n75) );
  XNOR2X1 U119 ( .A1(inp[7]), .A2(n113), .Y(n130) );
  INVX1 U120 ( .A(n130), .Y(n77) );
  AND2X1 U121 ( .A1(n81), .A2(n90), .Y(n135) );
  INVX1 U122 ( .A(n135), .Y() );
  INVX1 U123 ( .A(n144), .Y(n79) );
  AND2X1 U124 ( .A1(n143), .A2(n144), .Y(n145) );
  INVX1 U125 ( .A(n145), .Y(n80) );
  AND2X1 U126 ( .A1(n132), .A2(n133), .Y(n134) );
  INVX1 U127 ( .A(n134), .Y(n81) );
  AND2X1 U128 ( .A1(n104), .A2(n133), .Y(n107) );
  INVX1 U129 ( .A(n107), .Y(n82) );
  AND2X1 U130 ( .A1(n116), .A2(n115), .Y(n124) );
  INVX1 U131 ( .A(n124), .Y(n83) );
  AND2X1 U132 ( .A1(n139), .A2(inp[0]), .Y(n141) );
  INVX1 U133 ( .A(n141), .Y(n84) );
  AND2X1 U134 ( .A1(n108), .A2(inp[3]), .Y(n121) );
  INVX1 U135 ( .A(n121), .Y(n85) );
  XNOR2X1 U136 ( .A1(n71), .A2(n101), .Y(n105) );
  INVX1 U137 ( .A(n105), .Y(n86) );
  AND2X1 U138 ( .A1(n72), .A2(n71), .Y(n126) );
  INVX1 U139 ( .A(n126), .Y(n87) );
  OR2X1 U140 ( .A1(inp[14]), .A2(n98), .Y(n120) );
  INVX1 U141 ( .A(n120), .Y(n88) );
  AND2X1 U142 ( .A1(n100), .A2(inp[6]), .Y(n111) );
  INVX1 U143 ( .A(n111), .Y(n89) );
  OR2X1 U144 ( .A1(n83), .A2(n123), .Y(n125) );
  INVX1 U145 ( .A(n125), .Y(n90) );
  AND2X1 U146 ( .A1(n106), .A2(n82), .Y(n114) );
  INVX1 U147 ( .A(n114), .Y(n91) );
  AND2X1 U148 ( .A1(n68), .A2(n64), .Y(n112) );
  INVX1 U149 ( .A(n112), .Y(n92) );
  AND2X1 U150 ( .A1(n122), .A2(n85), .Y(n129) );
  INVX1 U151 ( .A(n129), .Y(n93) );
  XNOR2X1 U152 ( .A1(n120), .A2(n110), .Y(n131) );
  INVX1 U153 ( .A(n131), .Y(n94) );
  AND2X1 U154 ( .A1(n103), .A2(inp[12]), .Y(n142) );
  INVX1 U155 ( .A(n142), .Y(n95) );
  AND2X1 U156 ( .A1(n119), .A2(n118), .Y(n128) );
  INVX1 U157 ( .A(n128), .Y(n96) );
  AND2X1 U158 ( .A1(n92), .A2(inp[11]), .Y() );
  INVX1 U159 ( .A(1'b0), .Y(n97) );
  INVX1 U160 ( .A(inp[2]), .Y(n98) );
  INVX1 U161 ( .A(inp[6]), .Y(n99) );
  INVX1 U162 ( .A(inp[10]), .Y(n100) );
  NAND2X1 U163 ( .A1(n64), .A2(n89), .Y(n101) );
  INVX1 U164 ( .A(inp[9]), .Y(n102) );
  OR2X1 U165 ( .A1(n70), .A2(n86), .Y(n133) );
  INVX1 U166 ( .A(inp[0]), .Y(n103) );
  NOR2X1 U167 ( .A1(n87), .A2(n95), .Y(n104) );
  NAND2X1 U168 ( .A1(n70), .A2(n86), .Y(n106) );
  INVX1 U169 ( .A(inp[15]), .Y(n108) );
  INVX1 U170 ( .A(inp[3]), .Y(n109) );
  NAND2X1 U171 ( .A1(1'b1), .A2(n85), .Y(n110) );
  OR2X1 U172 ( .A1(n92), .A2(inp[11]), .Y(n118) );
  NAND2X1 U173 ( .A1(n118), .A2(n97), .Y(n113) );
  NAND2X1 U174 ( .A1(n73), .A2(n77), .Y(n116) );
  NAND2X1 U175 ( .A1(n131), .A2(n91), .Y(n115) );
  NAND2X1 U176 ( .A1(n97), .A2(inp[7]), .Y(n119) );
  NAND2X1 U177 ( .A1(1'b1), .A2(n88), .Y(n122) );
  NOR2X1 U178 ( .A1(n96), .A2(n93), .Y(n123) );
  NAND2X1 U179 ( .A1(n95), .A2(n87), .Y(n127) );
  NOR2X1 U180 ( .A1(n125), .A2(n127), .Y(n137) );
  NAND2X1 U181 ( .A1(n96), .A2(n93), .Y(n136) );
  NAND2X1 U182 ( .A1(n130), .A2(n94), .Y(n132) );
  INVX1 U183 ( .A(inp[1]), .Y(n138) );
  NOR2X1 U184 ( .A1(inp[13]), .A2(n138), .Y() );
  INVX1 U185 ( .A(inp[12]), .Y(n139) );
  AND2X1 U186 ( .A1(n79), .A2(n66), .Y(out[1]) );
  AND2X1 U187 ( .A1(n74), .A2(n90), .Y(n144) );
  NAND2X1 U188 ( .A1(n95), .A2(n84), .Y(n143) );
  NAND2X1 U189 ( .A1(n66), .A2(n80), .Y(out[0]) );
endmodule


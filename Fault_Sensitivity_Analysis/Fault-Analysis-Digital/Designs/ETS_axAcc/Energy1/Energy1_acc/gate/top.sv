/////////////////////////////////////////////////////////////
// Created by: Synopsys DC Ultra(TM) in wire load mode
// Version   : S-2021.06-SP2
// Date      : Thu Dec  7 21:35:59 2023
/////////////////////////////////////////////////////////////


module top ( inp, out );
  input [31:0] inp;
  output [1:0] out;
  wire   n49, n50, n51, n52, n53, n54, n56, n57, n58, n59, n60, n61, n62, n63,
         n64, n65, n66, n67, n68, n69, n70, n71, n72, n73, n74, n75, n76, n77,
         n78, n79, n80, n81, n82, n83, n84, n85, n86, n87, n88, n89, n90, n91,
         n92, n93, n94, n95, n96, n97, n98, n99, n100, n101, n102, n103, n104,
         n105, n106, n107, n108, n109, n110, n111, n112, n113, n114, n115,
         n116, n117, n118, n119, n120, n121, n122, n123, n124, n125, n126,
         n127, n128, n129, n130, n131, n132, n133, n134, n135, n136, n137,
         n138, n139, n140, n141, n142, n143, n144, n145, n146, n147, n148,
         n149, n150, n151, n152, n153, n154, n155, n156, n157;

  XOR2X1 U101 ( .A1(n53), .A2(inp[1]), .Y(n49) );
  NAND2X1 U102 ( .A1(n102), .A2(inp[24]), .Y(n50) );
  NAND2X1 U103 ( .A1(n88), .A2(inp[24]), .Y(n51) );
  NAND2X1 U104 ( .A1(n100), .A2(inp[5]), .Y(n52) );
  NAND2X1 U105 ( .A1(inp[16]), .A2(inp[8]), .Y(n53) );
  NOR2X1 U106 ( .A1(n91), .A2(n63), .Y(n54) );
  NOR2X1 U107 ( .A1(n117), .A2(n144), .Y(out[1]) );
  NAND2X1 U108 ( .A1(n61), .A2(n68), .Y(n56) );
  AND2X1 U109 ( .A1(n52), .A2(n138), .Y() );
  INVX1 U110 ( .A(1'b1), .Y(n57) );
  AND2X1 U111 ( .A1(n54), .A2(n145), .Y(n151) );
  INVX1 U112 ( .A(n151), .Y(n58) );
  AND2X1 U113 ( .A1(n90), .A2(n89), .Y(n62) );
  INVX1 U114 ( .A(n62), .Y(n59) );
  NAND2X1 U115 ( .A1(n109), .A2(n70), .Y(n60) );
  NAND2X1 U116 ( .A1(n125), .A2(n71), .Y(n61) );
  AND2X1 U117 ( .A1(1'b1), .A2(n117), .Y(n63) );
  NAND2X1 U118 ( .A1(n101), .A2(inp[17]), .Y(n64) );
  INVX1 U119 ( .A(n124), .Y(n65) );
  AND2X1 U120 ( .A1(n51), .A2(n142), .Y(n143) );
  INVX1 U121 ( .A(n143), .Y(n66) );
  AND2X1 U122 ( .A1(n50), .A2(n105), .Y(n106) );
  INVX1 U123 ( .A(n106), .Y(n67) );
  AND2X1 U124 ( .A1(n139), .A2(n59), .Y(n140) );
  INVX1 U125 ( .A(n140), .Y(n68) );
  AND2X1 U126 ( .A1(n121), .A2(n119), .Y(n122) );
  INVX1 U127 ( .A(n122), .Y(n69) );
  AND2X1 U128 ( .A1(n108), .A2(n104), .Y(n110) );
  INVX1 U129 ( .A(n110), .Y(n70) );
  AND2X1 U130 ( .A1(n98), .A2(1'b1), .Y(n126) );
  INVX1 U131 ( .A(n126), .Y(n71) );
  AND2X1 U132 ( .A1(n127), .A2(n128), .Y(n131) );
  INVX1 U133 ( .A(n131), .Y(n72) );
  AND2X1 U134 ( .A1(n133), .A2(n132), .Y(n135) );
  INVX1 U135 ( .A(n135), .Y(n73) );
  AND2X1 U136 ( .A1(n75), .A2(n155), .Y(n157) );
  INVX1 U137 ( .A(n157), .Y(n74) );
  AND2X1 U138 ( .A1(n149), .A2(n94), .Y(n156) );
  INVX1 U139 ( .A(n156), .Y(n75) );
  AND2X1 U140 ( .A1(n150), .A2(n138), .Y(n137) );
  INVX1 U141 ( .A(n137), .Y(n76) );
  AND2X1 U142 ( .A1(n81), .A2(1'b1), .Y(n118) );
  INVX1 U143 ( .A(n118), .Y(n77) );
  AND2X1 U144 ( .A1(n130), .A2(n129), .Y(n134) );
  INVX1 U145 ( .A(n134), .Y(n78) );
  AND2X1 U146 ( .A1(n93), .A2(n91), .Y(n147) );
  INVX1 U147 ( .A(n147), .Y(n79) );
  AND2X1 U148 ( .A1(n141), .A2(n66), .Y(n145) );
  INVX1 U149 ( .A(n145), .Y(n80) );
  AND2X1 U150 ( .A1(n64), .A2(n104), .Y(n102) );
  INVX1 U151 ( .A(n102), .Y(n81) );
  AND2X1 U152 ( .A1(n107), .A2(n67), .Y(n113) );
  INVX1 U153 ( .A(n113), .Y(n82) );
  AND2X1 U154 ( .A1(n50), .A2(n77), .Y(n120) );
  INVX1 U155 ( .A(n120), .Y(n83) );
  AND2X1 U156 ( .A1(inp[21]), .A2(inp[4]), .Y(n100) );
  INVX1 U157 ( .A(n100), .Y(n84) );
  AND2X1 U158 ( .A1(inp[11]), .A2(inp[3]), .Y() );
  INVX1 U159 ( .A(1'b0), .Y(n85) );
  AND2X1 U160 ( .A1(n60), .A2(n111), .Y(n119) );
  INVX1 U161 ( .A(n119), .Y() );
  AND2X1 U162 ( .A1(n84), .A2(n99), .Y(n146) );
  INVX1 U163 ( .A(n146), .Y(n87) );
  AND2X1 U164 ( .A1(n96), .A2(n85), .Y(n117) );
  INVX1 U165 ( .A(n117), .Y(n88) );
  AND2X1 U166 ( .A1(n73), .A2(n78), .Y(n138) );
  INVX1 U167 ( .A(n138), .Y(n89) );
  AND2X1 U168 ( .A1(n72), .A2(n78), .Y(n150) );
  INVX1 U169 ( .A(n150), .Y(n90) );
  AND2X1 U170 ( .A1(n115), .A2(n116), .Y(n141) );
  INVX1 U171 ( .A(n141), .Y(n91) );
  OR2X1 U172 ( .A1(n101), .A2(inp[17]), .Y(n104) );
  INVX1 U173 ( .A(n104), .Y(n92) );
  OR2X1 U174 ( .A1(n51), .A2(n142), .Y(n93) );
  NOR2X1 U175 ( .A1(n145), .A2(n54), .Y(n94) );
  NOR2X1 U176 ( .A1(inp[11]), .A2(inp[3]), .Y(n130) );
  INVX1 U177 ( .A(n130), .Y(n95) );
  NAND2X1 U178 ( .A1(n95), .A2(inp[19]), .Y(n96) );
  INVX1 U179 ( .A(inp[24]), .Y() );
  NAND2X1 U180 ( .A1(1'b0), .A2(inp[19]), .Y(n98) );
  INVX1 U181 ( .A(inp[5]), .Y(n99) );
  AND2X1 U182 ( .A1(n52), .A2(n87), .Y(n124) );
  INVX1 U183 ( .A(inp[13]), .Y(n101) );
  AND2X1 U184 ( .A1(n77), .A2(inp[1]), .Y(n103) );
  NAND2X1 U185 ( .A1(n103), .A2(n92), .Y(n107) );
  NOR2X1 U186 ( .A1(n92), .A2(n103), .Y(n105) );
  OR2X1 U187 ( .A1(inp[14]), .A2(inp[2]), .Y(n109) );
  INVX1 U188 ( .A(n109), .Y(n114) );
  NOR2X1 U189 ( .A1(n82), .A2(n114), .Y(n112) );
  NAND2X1 U190 ( .A1(inp[14]), .A2(inp[2]), .Y(n108) );
  INVX1 U191 ( .A(inp[11]), .Y(n111) );
  NOR2X1 U192 ( .A1(n112), .A2(1'b0), .Y(n116) );
  NAND2X1 U193 ( .A1(n114), .A2(n82), .Y(n115) );
  NOR2X1 U194 ( .A1(inp[1]), .A2(n83), .Y(n123) );
  NAND2X1 U195 ( .A1(n83), .A2(inp[1]), .Y(n121) );
  OR2X1 U196 ( .A1(n123), .A2(n69), .Y(n142) );
  NAND2X1 U197 ( .A1(n124), .A2(n79), .Y(n125) );
  OR2X1 U198 ( .A1(n49), .A2(inp[17]), .Y(n128) );
  NAND2X1 U199 ( .A1(n49), .A2(inp[17]), .Y(n127) );
  INVX1 U200 ( .A(inp[19]), .Y(n129) );
  NAND2X1 U201 ( .A1(n53), .A2(inp[8]), .Y(n133) );
  NAND2X1 U202 ( .A1(n53), .A2(inp[16]), .Y(n132) );
  NAND2X1 U203 ( .A1(n54), .A2(n76), .Y(n139) );
  NAND2X1 U204 ( .A1(n56), .A2(n80), .Y(n144) );
  NAND2X1 U205 ( .A1(n89), .A2(n65), .Y(n148) );
  NAND2X1 U206 ( .A1(n148), .A2(n79), .Y(n149) );
  NAND2X1 U207 ( .A1(n117), .A2(n90), .Y(n154) );
  NOR2X1 U208 ( .A1(n57), .A2(n58), .Y(n153) );
  OR2X1 U209 ( .A1(n154), .A2(n153), .Y(n155) );
  NOR2X1 U210 ( .A1(n74), .A2(out[1]), .Y(out[0]) );
endmodule

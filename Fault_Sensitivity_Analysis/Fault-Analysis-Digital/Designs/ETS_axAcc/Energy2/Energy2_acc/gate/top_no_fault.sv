/////////////////////////////////////////////////////////////
// Created by: Synopsys DC Ultra(TM) in wire load mode
// Version   : S-2021.06-SP2
// Date      : Thu Dec  7 21:36:57 2023
/////////////////////////////////////////////////////////////


module top ( inp, out );
  input [31:0] inp;
  output [1:0] out;
  wire   n56, n57, n58, n59, n60, n61, n62, n63, n64, n65, n66, n67, n68, n69,
         n70, n71, n72, n73, n74, n75, n76, n77, n78, n79, n80, n81, n82, n83,
         n84, n85, n86, n87, n88, n89, n90, n91, n92, n93, n94, n95, n96, n97,
         n98, n99, n100, n101, n102, n103, n104, n105, n106, n107, n108, n109,
         n110, n111, n112, n113, n114, n115, n116, n117, n118, n119, n120,
         n121;

  NAND2X1 U99 ( .A1(n73), .A2(inp[21]), .Y(n56) );
  NAND2X1 U100 ( .A1(n89), .A2(inp[13]), .Y(n57) );
  NAND2X1 U101 ( .A1(n120), .A2(n65), .Y(n58) );
  NAND2X1 U102 ( .A1(n96), .A2(n63), .Y(n59) );
  NOR2X1 U103 ( .A1(n102), .A2(n101), .Y(n60) );
  AND2X1 U104 ( .A1(n109), .A2(n60), .Y(n110) );
  INVX1 U105 ( .A(n110), .Y(n61) );
  AND2X1 U106 ( .A1(n64), .A2(n75), .Y(n121) );
  INVX1 U107 ( .A(n121), .Y(n62) );
  AND2X1 U108 ( .A1(n93), .A2(n92), .Y(n97) );
  INVX1 U109 ( .A(n97), .Y(n63) );
  AND2X1 U110 ( .A1(n118), .A2(n117), .Y(n119) );
  INVX1 U111 ( .A(n119), .Y(n64) );
  AND2X1 U112 ( .A1(n90), .A2(n74), .Y(n91) );
  INVX1 U113 ( .A(n91), .Y(n65) );
  AND2X1 U114 ( .A1(n86), .A2(n85), .Y(n88) );
  INVX1 U115 ( .A(n88), .Y(n66) );
  AND2X1 U116 ( .A1(n104), .A2(n73), .Y(n106) );
  INVX1 U117 ( .A(n106), .Y(n67) );
  AND2X1 U118 ( .A1(n114), .A2(n105), .Y(n81) );
  INVX1 U119 ( .A(n81), .Y(n68) );
  AND2X1 U120 ( .A1(n83), .A2(n82), .Y(n84) );
  INVX1 U121 ( .A(n84), .Y(n69) );
  AND2X1 U122 ( .A1(n99), .A2(n98), .Y(n100) );
  INVX1 U123 ( .A(n100), .Y(n70) );
  AND2X1 U124 ( .A1(n74), .A2(n86), .Y(n107) );
  INVX1 U125 ( .A(n107), .Y(n71) );
  AND2X1 U126 ( .A1(n102), .A2(n101), .Y(n112) );
  INVX1 U127 ( .A(n112), .Y(n72) );
  AND2X1 U128 ( .A1(inp[21]), .A2(inp[8]), .Y(n103) );
  INVX1 U129 ( .A(n103), .Y(n73) );
  AND2X1 U130 ( .A1(n78), .A2(n56), .Y(n89) );
  INVX1 U131 ( .A(n89), .Y(n74) );
  OR2X1 U132 ( .A1(n80), .A2(inp[15]), .Y(n120) );
  INVX1 U133 ( .A(n120), .Y(n75) );
  AND2X1 U134 ( .A1(n80), .A2(inp[15]), .Y(n105) );
  INVX1 U135 ( .A(n105), .Y(n76) );
  AND2X1 U136 ( .A1(n87), .A2(n66), .Y(n95) );
  INVX1 U137 ( .A(n95), .Y(n77) );
  NAND2X1 U138 ( .A1(n73), .A2(inp[8]), .Y(n78) );
  INVX1 U139 ( .A(inp[13]), .Y(n86) );
  INVX1 U140 ( .A(inp[19]), .Y(n80) );
  NOR2X1 U141 ( .A1(n71), .A2(n75), .Y(n79) );
  NAND2X1 U142 ( .A1(n79), .A2(inp[16]), .Y(n83) );
  NOR2X1 U143 ( .A1(n103), .A2(n57), .Y(n114) );
  NAND2X1 U144 ( .A1(n68), .A2(inp[25]), .Y(n82) );
  NAND2X1 U145 ( .A1(n69), .A2(inp[17]), .Y(n99) );
  INVX1 U146 ( .A(inp[25]), .Y(n85) );
  INVX1 U147 ( .A(inp[17]), .Y(n116) );
  NAND2X1 U148 ( .A1(n116), .A2(inp[13]), .Y(n87) );
  NAND2X1 U149 ( .A1(n89), .A2(n77), .Y(n93) );
  NAND2X1 U150 ( .A1(inp[13]), .A2(inp[25]), .Y(n90) );
  NOR2X1 U151 ( .A1(n105), .A2(n58), .Y(n92) );
  NAND2X1 U152 ( .A1(n103), .A2(inp[15]), .Y(n94) );
  OR2X1 U153 ( .A1(n77), .A2(n94), .Y(n96) );
  NAND2X1 U154 ( .A1(n59), .A2(inp[16]), .Y(n98) );
  INVX1 U155 ( .A(inp[18]), .Y(n102) );
  INVX1 U156 ( .A(inp[11]), .Y(n101) );
  NAND2X1 U157 ( .A1(n72), .A2(n70), .Y(n111) );
  NAND2X1 U158 ( .A1(n57), .A2(inp[17]), .Y(n104) );
  NOR2X1 U159 ( .A1(n67), .A2(n76), .Y(n108) );
  NAND2X1 U160 ( .A1(n108), .A2(n71), .Y(n109) );
  NAND2X1 U161 ( .A1(n111), .A2(n61), .Y(out[1]) );
  NAND2X1 U162 ( .A1(n72), .A2(inp[25]), .Y(n113) );
  NAND2X1 U163 ( .A1(n113), .A2(inp[17]), .Y(n118) );
  INVX1 U164 ( .A(n114), .Y(n115) );
  NAND2X1 U165 ( .A1(n116), .A2(n115), .Y(n117) );
  NOR2X1 U166 ( .A1(n60), .A2(n62), .Y(out[0]) );
endmodule


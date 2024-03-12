//generic design, we don't actually know the weights and intercepts, they are given to the network through the testbench.
//weights: [[[0, 0, 2, -3, -1, -4, -1, 0, -5, 1, 3, 1, -1, 1, -1, 2, -3, 2, -4, 2, -3], [34, -23, 4, -19, 11, -19, 41, 56, 19, 30, 35, -1, -30, -36, -73, 63, -89, -36, -50, 60, -12], [11, 68, -17, 16, 5, 33, 7, -30, -25, -25, 27, 58, 13, -52, -27, 1, 39, -93, -17, -22, -9]], [[-7, -45, 28], [6, -21, -22], [5, 95, -52]]]
//intercepts: [[-73, 748, 1077], [-1547, 902, -2131]]
module top (inp, weights, biases, out);
input [83:0] inp;
input [575:0] weights;
input [74:0] biases;
output [1:0] out;

// layer: 0 - neuron: 0
    //weight 0 : skip
    //weight abs(0) : 8'b00000000
    wire signed [12-1:0] n_0_0_po_0;
    assign n_0_0_po_0 = $signed({1'b0, inp[3:0]}) * $signed(weights[8-1:0]);

    //weight 0 : skip
    //weight abs(0) : 8'b00000000
    wire signed [12-1:0] n_0_0_po_1;
    assign n_0_0_po_1 = $signed({1'b0, inp[7:4]}) * $signed(weights[16-1:8]);

    //weight 2 : 8'b00000010
    wire signed [12-1:0] n_0_0_po_2;
    assign n_0_0_po_2 = $signed({1'b0, inp[11:8]}) * $signed(weights[24-1:16]);

    //weight abs(-3) : 8'b00000011
    wire signed [12-1:0] n_0_0_po_3;
    assign n_0_0_po_3 = $signed({1'b0, inp[15:12]}) * $signed(weights[32-1:24]);

    //weight abs(-1) : 8'b00000001
    wire signed [12-1:0] n_0_0_po_4;
    assign n_0_0_po_4 = $signed({1'b0, inp[19:16]}) * $signed(weights[40-1:32]);

    //weight abs(-4) : 8'b00000100
    wire signed [12-1:0] n_0_0_po_5;
    assign n_0_0_po_5 = $signed({1'b0, inp[23:20]}) * $signed(weights[48-1:40]);

    //weight abs(-1) : 8'b00000001
    wire signed [12-1:0] n_0_0_po_6;
    assign n_0_0_po_6 = $signed({1'b0, inp[27:24]}) * $signed(weights[56-1:48]);

    //weight 0 : skip
    //weight abs(0) : 8'b00000000
    wire signed [12-1:0] n_0_0_po_7;
    assign n_0_0_po_7 = $signed({1'b0, inp[31:28]}) * $signed(weights[64-1:56]);

    //weight abs(-5) : 8'b00000101
    wire signed [12-1:0] n_0_0_po_8;
    assign n_0_0_po_8 = $signed({1'b0, inp[35:32]}) * $signed(weights[72-1:64]);

    //weight 1 : 8'b00000001
    wire signed [12-1:0] n_0_0_po_9;
    assign n_0_0_po_9 = $signed({1'b0, inp[39:36]}) * $signed(weights[80-1:72]);

    //weight 3 : 8'b00000011
    wire signed [12-1:0] n_0_0_po_10;
    assign n_0_0_po_10 = $signed({1'b0, inp[43:40]}) * $signed(weights[88-1:80]);

    //weight 1 : 8'b00000001
    wire signed [12-1:0] n_0_0_po_11;
    assign n_0_0_po_11 = $signed({1'b0, inp[47:44]}) * $signed(weights[96-1:88]);

    //weight abs(-1) : 8'b00000001
    wire signed [12-1:0] n_0_0_po_12;
    assign n_0_0_po_12 = $signed({1'b0, inp[51:48]}) * $signed(weights[104-1:96]);

    //weight 1 : 8'b00000001
    wire signed [12-1:0] n_0_0_po_13;
    assign n_0_0_po_13 = $signed({1'b0, inp[55:52]}) * $signed(weights[112-1:104]);

    //weight abs(-1) : 8'b00000001
    wire signed [12-1:0] n_0_0_po_14;
    assign n_0_0_po_14 = $signed({1'b0, inp[59:56]}) * $signed(weights[120-1:112]);

    //weight 2 : 8'b00000010
    wire signed [12-1:0] n_0_0_po_15;
    assign n_0_0_po_15 = $signed({1'b0, inp[63:60]}) * $signed(weights[128-1:120]);

    //weight abs(-3) : 8'b00000011
    wire signed [12-1:0] n_0_0_po_16;
    assign n_0_0_po_16 = $signed({1'b0, inp[67:64]}) * $signed(weights[136-1:128]);

    //weight 2 : 8'b00000010
    wire signed [12-1:0] n_0_0_po_17;
    assign n_0_0_po_17 = $signed({1'b0, inp[71:68]}) * $signed(weights[144-1:136]);

    //weight abs(-4) : 8'b00000100
    wire signed [12-1:0] n_0_0_po_18;
    assign n_0_0_po_18 = $signed({1'b0, inp[75:72]}) * $signed(weights[152-1:144]);

    //weight 2 : 8'b00000010
    wire signed [12-1:0] n_0_0_po_19;
    assign n_0_0_po_19 = $signed({1'b0, inp[79:76]}) * $signed(weights[160-1:152]);

    //weight abs(-3) : 8'b00000011
    wire signed [12-1:0] n_0_0_po_20;
    assign n_0_0_po_20 = $signed({1'b0, inp[83:80]}) * $signed(weights[168-1:160]);

    //accumulate products
    wire signed [32:0] n_0_0_sum;
    assign n_0_0_sum = $signed(biases[12-1:0]) + $signed(n_0_0_po_0) + $signed(n_0_0_po_1) + $signed(n_0_0_po_2) + $signed(n_0_0_po_3) + $signed(n_0_0_po_4) + $signed(n_0_0_po_5) + $signed(n_0_0_po_6) + $signed(n_0_0_po_7) + $signed(n_0_0_po_8) + $signed(n_0_0_po_9) + $signed(n_0_0_po_10) + $signed(n_0_0_po_11) + $signed(n_0_0_po_12) + $signed(n_0_0_po_13) + $signed(n_0_0_po_14) + $signed(n_0_0_po_15) + $signed(n_0_0_po_16) + $signed(n_0_0_po_17) + $signed(n_0_0_po_18) + $signed(n_0_0_po_19) + $signed(n_0_0_po_20);
    //relu
    wire [31:0] n_0_0;
    assign n_0_0 = (n_0_0_sum<0) ? $unsigned({32{1'b0}}) : $unsigned(n_0_0_sum);

// layer: 0 - neuron: 1
    //weight 34 : 8'b00100010
    wire signed [12-1:0] n_0_1_po_0;
    assign n_0_1_po_0 = $signed({1'b0, inp[3:0]}) * $signed(weights[176-1:168]);

    //weight abs(-23) : 8'b00010111
    wire signed [12-1:0] n_0_1_po_1;
    assign n_0_1_po_1 = $signed({1'b0, inp[7:4]}) * $signed(weights[184-1:176]);

    //weight 4 : 8'b00000100
    wire signed [12-1:0] n_0_1_po_2;
    assign n_0_1_po_2 = $signed({1'b0, inp[11:8]}) * $signed(weights[192-1:184]);

    //weight abs(-19) : 8'b00010011
    wire signed [12-1:0] n_0_1_po_3;
    assign n_0_1_po_3 = $signed({1'b0, inp[15:12]}) * $signed(weights[200-1:192]);

    //weight 11 : 8'b00001011
    wire signed [12-1:0] n_0_1_po_4;
    assign n_0_1_po_4 = $signed({1'b0, inp[19:16]}) * $signed(weights[208-1:200]);

    //weight abs(-19) : 8'b00010011
    wire signed [12-1:0] n_0_1_po_5;
    assign n_0_1_po_5 = $signed({1'b0, inp[23:20]}) * $signed(weights[216-1:208]);

    //weight 41 : 8'b00101001
    wire signed [12-1:0] n_0_1_po_6;
    assign n_0_1_po_6 = $signed({1'b0, inp[27:24]}) * $signed(weights[224-1:216]);

    //weight 56 : 8'b00111000
    wire signed [12-1:0] n_0_1_po_7;
    assign n_0_1_po_7 = $signed({1'b0, inp[31:28]}) * $signed(weights[232-1:224]);

    //weight 19 : 8'b00010011
    wire signed [12-1:0] n_0_1_po_8;
    assign n_0_1_po_8 = $signed({1'b0, inp[35:32]}) * $signed(weights[240-1:232]);

    //weight 30 : 8'b00011110
    wire signed [12-1:0] n_0_1_po_9;
    assign n_0_1_po_9 = $signed({1'b0, inp[39:36]}) * $signed(weights[248-1:240]);

    //weight 35 : 8'b00100011
    wire signed [12-1:0] n_0_1_po_10;
    assign n_0_1_po_10 = $signed({1'b0, inp[43:40]}) * $signed(weights[256-1:248]);

    //weight abs(-1) : 8'b00000001
    wire signed [12-1:0] n_0_1_po_11;
    assign n_0_1_po_11 = $signed({1'b0, inp[47:44]}) * $signed(weights[264-1:256]);

    //weight abs(-30) : 8'b00011110
    wire signed [12-1:0] n_0_1_po_12;
    assign n_0_1_po_12 = $signed({1'b0, inp[51:48]}) * $signed(weights[272-1:264]);

    //weight abs(-36) : 8'b00100100
    wire signed [12-1:0] n_0_1_po_13;
    assign n_0_1_po_13 = $signed({1'b0, inp[55:52]}) * $signed(weights[280-1:272]);

    //weight abs(-73) : 8'b01001001
    wire signed [12-1:0] n_0_1_po_14;
    assign n_0_1_po_14 = $signed({1'b0, inp[59:56]}) * $signed(weights[288-1:280]);

    //weight 63 : 8'b00111111
    wire signed [12-1:0] n_0_1_po_15;
    assign n_0_1_po_15 = $signed({1'b0, inp[63:60]}) * $signed(weights[296-1:288]);

    //weight abs(-89) : 8'b01011001
    wire signed [12-1:0] n_0_1_po_16;
    assign n_0_1_po_16 = $signed({1'b0, inp[67:64]}) * $signed(weights[304-1:296]);

    //weight abs(-36) : 8'b00100100
    wire signed [12-1:0] n_0_1_po_17;
    assign n_0_1_po_17 = $signed({1'b0, inp[71:68]}) * $signed(weights[312-1:304]);

    //weight abs(-50) : 8'b00110010
    wire signed [12-1:0] n_0_1_po_18;
    assign n_0_1_po_18 = $signed({1'b0, inp[75:72]}) * $signed(weights[320-1:312]);

    //weight 60 : 8'b00111100
    wire signed [12-1:0] n_0_1_po_19;
    assign n_0_1_po_19 = $signed({1'b0, inp[79:76]}) * $signed(weights[328-1:320]);

    //weight abs(-12) : 8'b00001100
    wire signed [12-1:0] n_0_1_po_20;
    assign n_0_1_po_20 = $signed({1'b0, inp[83:80]}) * $signed(weights[336-1:328]);

    //accumulate products
    wire signed [32:0] n_0_1_sum;
    assign n_0_1_sum = $signed(biases[24-1:12]) + $signed(n_0_1_po_0) + $signed(n_0_1_po_1) + $signed(n_0_1_po_2) + $signed(n_0_1_po_3) + $signed(n_0_1_po_4) + $signed(n_0_1_po_5) + $signed(n_0_1_po_6) + $signed(n_0_1_po_7) + $signed(n_0_1_po_8) + $signed(n_0_1_po_9) + $signed(n_0_1_po_10) + $signed(n_0_1_po_11) + $signed(n_0_1_po_12) + $signed(n_0_1_po_13) + $signed(n_0_1_po_14) + $signed(n_0_1_po_15) + $signed(n_0_1_po_16) + $signed(n_0_1_po_17) + $signed(n_0_1_po_18) + $signed(n_0_1_po_19) + $signed(n_0_1_po_20);
    //relu
    wire [31:0] n_0_1;
    assign n_0_1 = (n_0_1_sum<0) ? $unsigned({32{1'b0}}) : $unsigned(n_0_1_sum);

// layer: 0 - neuron: 2
    //weight 11 : 8'b00001011
    wire signed [12-1:0] n_0_2_po_0;
    assign n_0_2_po_0 = $signed({1'b0, inp[3:0]}) * $signed(weights[344-1:336]);

    //weight 68 : 8'b01000100
    wire signed [12-1:0] n_0_2_po_1;
    assign n_0_2_po_1 = $signed({1'b0, inp[7:4]}) * $signed(weights[352-1:344]);

    //weight abs(-17) : 8'b00010001
    wire signed [12-1:0] n_0_2_po_2;
    assign n_0_2_po_2 = $signed({1'b0, inp[11:8]}) * $signed(weights[360-1:352]);

    //weight 16 : 8'b00010000
    wire signed [12-1:0] n_0_2_po_3;
    assign n_0_2_po_3 = $signed({1'b0, inp[15:12]}) * $signed(weights[368-1:360]);

    //weight 5 : 8'b00000101
    wire signed [12-1:0] n_0_2_po_4;
    assign n_0_2_po_4 = $signed({1'b0, inp[19:16]}) * $signed(weights[376-1:368]);

    //weight 33 : 8'b00100001
    wire signed [12-1:0] n_0_2_po_5;
    assign n_0_2_po_5 = $signed({1'b0, inp[23:20]}) * $signed(weights[384-1:376]);

    //weight 7 : 8'b00000111
    wire signed [12-1:0] n_0_2_po_6;
    assign n_0_2_po_6 = $signed({1'b0, inp[27:24]}) * $signed(weights[392-1:384]);

    //weight abs(-30) : 8'b00011110
    wire signed [12-1:0] n_0_2_po_7;
    assign n_0_2_po_7 = $signed({1'b0, inp[31:28]}) * $signed(weights[400-1:392]);

    //weight abs(-25) : 8'b00011001
    wire signed [12-1:0] n_0_2_po_8;
    assign n_0_2_po_8 = $signed({1'b0, inp[35:32]}) * $signed(weights[408-1:400]);

    //weight abs(-25) : 8'b00011001
    wire signed [12-1:0] n_0_2_po_9;
    assign n_0_2_po_9 = $signed({1'b0, inp[39:36]}) * $signed(weights[416-1:408]);

    //weight 27 : 8'b00011011
    wire signed [12-1:0] n_0_2_po_10;
    assign n_0_2_po_10 = $signed({1'b0, inp[43:40]}) * $signed(weights[424-1:416]);

    //weight 58 : 8'b00111010
    wire signed [12-1:0] n_0_2_po_11;
    assign n_0_2_po_11 = $signed({1'b0, inp[47:44]}) * $signed(weights[432-1:424]);

    //weight 13 : 8'b00001101
    wire signed [12-1:0] n_0_2_po_12;
    assign n_0_2_po_12 = $signed({1'b0, inp[51:48]}) * $signed(weights[440-1:432]);

    //weight abs(-52) : 8'b00110100
    wire signed [12-1:0] n_0_2_po_13;
    assign n_0_2_po_13 = $signed({1'b0, inp[55:52]}) * $signed(weights[448-1:440]);

    //weight abs(-27) : 8'b00011011
    wire signed [12-1:0] n_0_2_po_14;
    assign n_0_2_po_14 = $signed({1'b0, inp[59:56]}) * $signed(weights[456-1:448]);

    //weight 1 : 8'b00000001
    wire signed [12-1:0] n_0_2_po_15;
    assign n_0_2_po_15 = $signed({1'b0, inp[63:60]}) * $signed(weights[464-1:456]);

    //weight 39 : 8'b00100111
    wire signed [12-1:0] n_0_2_po_16;
    assign n_0_2_po_16 = $signed({1'b0, inp[67:64]}) * $signed(weights[472-1:464]);

    //weight abs(-93) : 8'b01011101
    wire signed [12-1:0] n_0_2_po_17;
    assign n_0_2_po_17 = $signed({1'b0, inp[71:68]}) * $signed(weights[480-1:472]);

    //weight abs(-17) : 8'b00010001
    wire signed [12-1:0] n_0_2_po_18;
    assign n_0_2_po_18 = $signed({1'b0, inp[75:72]}) * $signed(weights[488-1:480]);

    //weight abs(-22) : 8'b00010110
    wire signed [12-1:0] n_0_2_po_19;
    assign n_0_2_po_19 = $signed({1'b0, inp[79:76]}) * $signed(weights[496-1:488]);

    //weight abs(-9) : 8'b00001001
    wire signed [12-1:0] n_0_2_po_20;
    assign n_0_2_po_20 = $signed({1'b0, inp[83:80]}) * $signed(weights[504-1:496]);

    //accumulate products
    wire signed [32:0] n_0_2_sum;
    assign n_0_2_sum = $signed(biases[36-1:24]) + $signed(n_0_2_po_0) + $signed(n_0_2_po_1) + $signed(n_0_2_po_2) + $signed(n_0_2_po_3) + $signed(n_0_2_po_4) + $signed(n_0_2_po_5) + $signed(n_0_2_po_6) + $signed(n_0_2_po_7) + $signed(n_0_2_po_8) + $signed(n_0_2_po_9) + $signed(n_0_2_po_10) + $signed(n_0_2_po_11) + $signed(n_0_2_po_12) + $signed(n_0_2_po_13) + $signed(n_0_2_po_14) + $signed(n_0_2_po_15) + $signed(n_0_2_po_16) + $signed(n_0_2_po_17) + $signed(n_0_2_po_18) + $signed(n_0_2_po_19) + $signed(n_0_2_po_20);
    //relu
    wire [31:0] n_0_2;
    assign n_0_2 = (n_0_2_sum<0) ? $unsigned({32{1'b0}}) : $unsigned(n_0_2_sum);

// layer: 1 - neuron: 0
    //weight abs(-7) : 8'b00000111
    wire signed [23-1:0] n_1_0_po_0;
    assign n_1_0_po_0 = $signed({1'b0, n_0_0}) * $signed(weights[512-1:504]);

    //weight abs(-45) : 8'b00101101
    wire signed [23-1:0] n_1_0_po_1;
    assign n_1_0_po_1 = $signed({1'b0, n_0_1}) * $signed(weights[520-1:512]);

    //weight 28 : 8'b00011100
    wire signed [23-1:0] n_1_0_po_2;
    assign n_1_0_po_2 = $signed({1'b0, n_0_2}) * $signed(weights[528-1:520]);

    //accumulate products
    wire signed [25:0] n_1_0_sum;
    assign n_1_0_sum = $signed(biases[49-1:36]) + $signed(n_1_0_po_0) + $signed(n_1_0_po_1) + $signed(n_1_0_po_2);
    //relu
    wire [24:0] n_1_0;
    assign n_1_0 = (n_1_0_sum<0) ? $unsigned({25{1'b0}}) : $unsigned(n_1_0_sum);

// layer: 1 - neuron: 1
    //weight 6 : 8'b00000110
    wire signed [23-1:0] n_1_1_po_0;
    assign n_1_1_po_0 = $signed({1'b0, n_0_0}) * $signed(weights[536-1:528]);

    //weight abs(-21) : 8'b00010101
    wire signed [23-1:0] n_1_1_po_1;
    assign n_1_1_po_1 = $signed({1'b0, n_0_1}) * $signed(weights[544-1:536]);

    //weight abs(-22) : 8'b00010110
    wire signed [23-1:0] n_1_1_po_2;
    assign n_1_1_po_2 = $signed({1'b0, n_0_2}) * $signed(weights[552-1:544]);

    //accumulate products
    wire signed [25:0] n_1_1_sum;
    assign n_1_1_sum = $signed(biases[62-1:49]) + $signed(n_1_1_po_0) + $signed(n_1_1_po_1) + $signed(n_1_1_po_2);
    //relu
    wire [24:0] n_1_1;
    assign n_1_1 = (n_1_1_sum<0) ? $unsigned({25{1'b0}}) : $unsigned(n_1_1_sum);

// layer: 1 - neuron: 2
    //weight 5 : 8'b00000101
    wire signed [23-1:0] n_1_2_po_0;
    assign n_1_2_po_0 = $signed({1'b0, n_0_0}) * $signed(weights[560-1:552]);

    //weight 95 : 8'b01011111
    wire signed [23-1:0] n_1_2_po_1;
    assign n_1_2_po_1 = $signed({1'b0, n_0_1}) * $signed(weights[568-1:560]);

    //weight abs(-52) : 8'b00110100
    wire signed [23-1:0] n_1_2_po_2;
    assign n_1_2_po_2 = $signed({1'b0, n_0_2}) * $signed(weights[576-1:568]);

    //accumulate products
    wire signed [25:0] n_1_2_sum;
    assign n_1_2_sum = $signed(biases[75-1:62]) + $signed(n_1_2_po_0) + $signed(n_1_2_po_1) + $signed(n_1_2_po_2);
    //relu
    wire [24:0] n_1_2;
    assign n_1_2 = (n_1_2_sum<0) ? $unsigned({25{1'b0}}) : $unsigned(n_1_2_sum);

// argmax: 3 classes, need 2 bits
// argmax inp: n_1_0, n_1_1, n_1_2
    //comp level 0
    wire cmp_0_0;
    wire [31:0] argmax_val_0_0;
    wire [1:0] argmax_idx_0_0;
    assign {cmp_0_0} = ( n_1_0 >= n_1_1 );
    assign {argmax_val_0_0} = ( cmp_0_0 ) ? n_1_0 : n_1_1;
    assign {argmax_idx_0_0} = ( cmp_0_0 ) ? 2'b00 : 2'b01;

    //comp level 1
    wire cmp_1_0;
    wire [31:0] argmax_val_1_0;
    wire [1:0] argmax_idx_1_0;
    assign {cmp_1_0} = ( argmax_val_0_0 >= n_1_2 );
    assign {argmax_val_1_0} = ( cmp_1_0 ) ? argmax_val_0_0 : n_1_2;
    assign {argmax_idx_1_0} = ( cmp_1_0 ) ? argmax_idx_0_0 : 2'b10;

    assign out = argmax_idx_1_0;

endmodule

//generic design, we don't actually know the weights and intercepts, they are given to the network through the testbench.
//weights: [[[16, 32, 18, -41, -44, 2, 0, 10, -12, -23, 36, 0, 6, 37, -13, 31], [-14, 12, -5, 17, 27, -6, -6, -26, -7, 13, 9, -17, -4, 16, 0, 79], [12, 0, 19, -5, -16, 26, -16, 20, 23, -14, 7, 24, 7, -31, -11, -53]], [[7, -4, -19], [-32, -32, 0], [-36, -17, -32], [-45, 10, 16], [9, -73, -7], [-2, 9, 11], [12, -17, -11], [-17, 7, -4], [0, 1, -30], [-2, -5, 9]]]
//intercepts: [[22, -192, -29], [-2150, 1446, 8870, -10964, 551, -11078, -1215, -3588, -2146, -7548]]
module top (inp, weights, biases, out);
input [63:0] inp;
input [623:0] weights;
input [176:0] biases;
output [3:0] out;

// layer: 0 - neuron: 0
    //weight 16 : 8'b00010000
    wire signed [12-1:0] n_0_0_po_0;
    assign n_0_0_po_0 = $signed({1'b0, inp[3:0]}) * $signed(weights[8-1:0]);

    //weight 32 : 8'b00100000
    wire signed [12-1:0] n_0_0_po_1;
    assign n_0_0_po_1 = $signed({1'b0, inp[7:4]}) * $signed(weights[16-1:8]);

    //weight 18 : 8'b00010010
    wire signed [12-1:0] n_0_0_po_2;
    assign n_0_0_po_2 = $signed({1'b0, inp[11:8]}) * $signed(weights[24-1:16]);

    //weight abs(-41) : 8'b00101001
    wire signed [12-1:0] n_0_0_po_3;
    assign n_0_0_po_3 = $signed({1'b0, inp[15:12]}) * $signed(weights[32-1:24]);

    //weight abs(-44) : 8'b00101100
    wire signed [12-1:0] n_0_0_po_4;
    assign n_0_0_po_4 = $signed({1'b0, inp[19:16]}) * $signed(weights[40-1:32]);

    //weight 2 : 8'b00000010
    wire signed [12-1:0] n_0_0_po_5;
    assign n_0_0_po_5 = $signed({1'b0, inp[23:20]}) * $signed(weights[48-1:40]);

    //weight 0 : skip
    //weight abs(0) : 8'b00000000
    wire signed [12-1:0] n_0_0_po_6;
    assign n_0_0_po_6 = $signed({1'b0, inp[27:24]}) * $signed(weights[56-1:48]);

    //weight 10 : 8'b00001010
    wire signed [12-1:0] n_0_0_po_7;
    assign n_0_0_po_7 = $signed({1'b0, inp[31:28]}) * $signed(weights[64-1:56]);

    //weight abs(-12) : 8'b00001100
    wire signed [12-1:0] n_0_0_po_8;
    assign n_0_0_po_8 = $signed({1'b0, inp[35:32]}) * $signed(weights[72-1:64]);

    //weight abs(-23) : 8'b00010111
    wire signed [12-1:0] n_0_0_po_9;
    assign n_0_0_po_9 = $signed({1'b0, inp[39:36]}) * $signed(weights[80-1:72]);

    //weight 36 : 8'b00100100
    wire signed [12-1:0] n_0_0_po_10;
    assign n_0_0_po_10 = $signed({1'b0, inp[43:40]}) * $signed(weights[88-1:80]);

    //weight 0 : skip
    //weight abs(0) : 8'b00000000
    wire signed [12-1:0] n_0_0_po_11;
    assign n_0_0_po_11 = $signed({1'b0, inp[47:44]}) * $signed(weights[96-1:88]);

    //weight 6 : 8'b00000110
    wire signed [12-1:0] n_0_0_po_12;
    assign n_0_0_po_12 = $signed({1'b0, inp[51:48]}) * $signed(weights[104-1:96]);

    //weight 37 : 8'b00100101
    wire signed [12-1:0] n_0_0_po_13;
    assign n_0_0_po_13 = $signed({1'b0, inp[55:52]}) * $signed(weights[112-1:104]);

    //weight abs(-13) : 8'b00001101
    wire signed [12-1:0] n_0_0_po_14;
    assign n_0_0_po_14 = $signed({1'b0, inp[59:56]}) * $signed(weights[120-1:112]);

    //weight 31 : 8'b00011111
    wire signed [12-1:0] n_0_0_po_15;
    assign n_0_0_po_15 = $signed({1'b0, inp[63:60]}) * $signed(weights[128-1:120]);

    //accumulate products
    wire signed [27:0] n_0_0_sum;
    assign n_0_0_sum = $signed(biases[9-1:0]) + $signed(n_0_0_po_0) + $signed(n_0_0_po_1) + $signed(n_0_0_po_2) + $signed(n_0_0_po_3) + $signed(n_0_0_po_4) + $signed(n_0_0_po_5) + $signed(n_0_0_po_6) + $signed(n_0_0_po_7) + $signed(n_0_0_po_8) + $signed(n_0_0_po_9) + $signed(n_0_0_po_10) + $signed(n_0_0_po_11) + $signed(n_0_0_po_12) + $signed(n_0_0_po_13) + $signed(n_0_0_po_14) + $signed(n_0_0_po_15);
    //relu
    wire [26:0] n_0_0;
    assign n_0_0 = (n_0_0_sum<0) ? $unsigned({27{1'b0}}) : $unsigned(n_0_0_sum);

// layer: 0 - neuron: 1
    //weight abs(-14) : 8'b00001110
    wire signed [12-1:0] n_0_1_po_0;
    assign n_0_1_po_0 = $signed({1'b0, inp[3:0]}) * $signed(weights[136-1:128]);

    //weight 12 : 8'b00001100
    wire signed [12-1:0] n_0_1_po_1;
    assign n_0_1_po_1 = $signed({1'b0, inp[7:4]}) * $signed(weights[144-1:136]);

    //weight abs(-5) : 8'b00000101
    wire signed [12-1:0] n_0_1_po_2;
    assign n_0_1_po_2 = $signed({1'b0, inp[11:8]}) * $signed(weights[152-1:144]);

    //weight 17 : 8'b00010001
    wire signed [12-1:0] n_0_1_po_3;
    assign n_0_1_po_3 = $signed({1'b0, inp[15:12]}) * $signed(weights[160-1:152]);

    //weight 27 : 8'b00011011
    wire signed [12-1:0] n_0_1_po_4;
    assign n_0_1_po_4 = $signed({1'b0, inp[19:16]}) * $signed(weights[168-1:160]);

    //weight abs(-6) : 8'b00000110
    wire signed [12-1:0] n_0_1_po_5;
    assign n_0_1_po_5 = $signed({1'b0, inp[23:20]}) * $signed(weights[176-1:168]);

    //weight abs(-6) : 8'b00000110
    wire signed [12-1:0] n_0_1_po_6;
    assign n_0_1_po_6 = $signed({1'b0, inp[27:24]}) * $signed(weights[184-1:176]);

    //weight abs(-26) : 8'b00011010
    wire signed [12-1:0] n_0_1_po_7;
    assign n_0_1_po_7 = $signed({1'b0, inp[31:28]}) * $signed(weights[192-1:184]);

    //weight abs(-7) : 8'b00000111
    wire signed [12-1:0] n_0_1_po_8;
    assign n_0_1_po_8 = $signed({1'b0, inp[35:32]}) * $signed(weights[200-1:192]);

    //weight 13 : 8'b00001101
    wire signed [12-1:0] n_0_1_po_9;
    assign n_0_1_po_9 = $signed({1'b0, inp[39:36]}) * $signed(weights[208-1:200]);

    //weight 9 : 8'b00001001
    wire signed [12-1:0] n_0_1_po_10;
    assign n_0_1_po_10 = $signed({1'b0, inp[43:40]}) * $signed(weights[216-1:208]);

    //weight abs(-17) : 8'b00010001
    wire signed [12-1:0] n_0_1_po_11;
    assign n_0_1_po_11 = $signed({1'b0, inp[47:44]}) * $signed(weights[224-1:216]);

    //weight abs(-4) : 8'b00000100
    wire signed [12-1:0] n_0_1_po_12;
    assign n_0_1_po_12 = $signed({1'b0, inp[51:48]}) * $signed(weights[232-1:224]);

    //weight 16 : 8'b00010000
    wire signed [12-1:0] n_0_1_po_13;
    assign n_0_1_po_13 = $signed({1'b0, inp[55:52]}) * $signed(weights[240-1:232]);

    //weight 0 : skip
    //weight abs(0) : 8'b00000000
    wire signed [12-1:0] n_0_1_po_14;
    assign n_0_1_po_14 = $signed({1'b0, inp[59:56]}) * $signed(weights[248-1:240]);

    //weight 79 : 8'b01001111
    wire signed [12-1:0] n_0_1_po_15;
    assign n_0_1_po_15 = $signed({1'b0, inp[63:60]}) * $signed(weights[256-1:248]);

    //accumulate products
    wire signed [27:0] n_0_1_sum;
    assign n_0_1_sum = $signed(biases[18-1:9]) + $signed(n_0_1_po_0) + $signed(n_0_1_po_1) + $signed(n_0_1_po_2) + $signed(n_0_1_po_3) + $signed(n_0_1_po_4) + $signed(n_0_1_po_5) + $signed(n_0_1_po_6) + $signed(n_0_1_po_7) + $signed(n_0_1_po_8) + $signed(n_0_1_po_9) + $signed(n_0_1_po_10) + $signed(n_0_1_po_11) + $signed(n_0_1_po_12) + $signed(n_0_1_po_13) + $signed(n_0_1_po_14) + $signed(n_0_1_po_15);
    //relu
    wire [26:0] n_0_1;
    assign n_0_1 = (n_0_1_sum<0) ? $unsigned({27{1'b0}}) : $unsigned(n_0_1_sum);

// layer: 0 - neuron: 2
    //weight 12 : 8'b00001100
    wire signed [12-1:0] n_0_2_po_0;
    assign n_0_2_po_0 = $signed({1'b0, inp[3:0]}) * $signed(weights[264-1:256]);

    //weight 0 : skip
    //weight abs(0) : 8'b00000000
    wire signed [12-1:0] n_0_2_po_1;
    assign n_0_2_po_1 = $signed({1'b0, inp[7:4]}) * $signed(weights[272-1:264]);

    //weight 19 : 8'b00010011
    wire signed [12-1:0] n_0_2_po_2;
    assign n_0_2_po_2 = $signed({1'b0, inp[11:8]}) * $signed(weights[280-1:272]);

    //weight abs(-5) : 8'b00000101
    wire signed [12-1:0] n_0_2_po_3;
    assign n_0_2_po_3 = $signed({1'b0, inp[15:12]}) * $signed(weights[288-1:280]);

    //weight abs(-16) : 8'b00010000
    wire signed [12-1:0] n_0_2_po_4;
    assign n_0_2_po_4 = $signed({1'b0, inp[19:16]}) * $signed(weights[296-1:288]);

    //weight 26 : 8'b00011010
    wire signed [12-1:0] n_0_2_po_5;
    assign n_0_2_po_5 = $signed({1'b0, inp[23:20]}) * $signed(weights[304-1:296]);

    //weight abs(-16) : 8'b00010000
    wire signed [12-1:0] n_0_2_po_6;
    assign n_0_2_po_6 = $signed({1'b0, inp[27:24]}) * $signed(weights[312-1:304]);

    //weight 20 : 8'b00010100
    wire signed [12-1:0] n_0_2_po_7;
    assign n_0_2_po_7 = $signed({1'b0, inp[31:28]}) * $signed(weights[320-1:312]);

    //weight 23 : 8'b00010111
    wire signed [12-1:0] n_0_2_po_8;
    assign n_0_2_po_8 = $signed({1'b0, inp[35:32]}) * $signed(weights[328-1:320]);

    //weight abs(-14) : 8'b00001110
    wire signed [12-1:0] n_0_2_po_9;
    assign n_0_2_po_9 = $signed({1'b0, inp[39:36]}) * $signed(weights[336-1:328]);

    //weight 7 : 8'b00000111
    wire signed [12-1:0] n_0_2_po_10;
    assign n_0_2_po_10 = $signed({1'b0, inp[43:40]}) * $signed(weights[344-1:336]);

    //weight 24 : 8'b00011000
    wire signed [12-1:0] n_0_2_po_11;
    assign n_0_2_po_11 = $signed({1'b0, inp[47:44]}) * $signed(weights[352-1:344]);

    //weight 7 : 8'b00000111
    wire signed [12-1:0] n_0_2_po_12;
    assign n_0_2_po_12 = $signed({1'b0, inp[51:48]}) * $signed(weights[360-1:352]);

    //weight abs(-31) : 8'b00011111
    wire signed [12-1:0] n_0_2_po_13;
    assign n_0_2_po_13 = $signed({1'b0, inp[55:52]}) * $signed(weights[368-1:360]);

    //weight abs(-11) : 8'b00001011
    wire signed [12-1:0] n_0_2_po_14;
    assign n_0_2_po_14 = $signed({1'b0, inp[59:56]}) * $signed(weights[376-1:368]);

    //weight abs(-53) : 8'b00110101
    wire signed [12-1:0] n_0_2_po_15;
    assign n_0_2_po_15 = $signed({1'b0, inp[63:60]}) * $signed(weights[384-1:376]);

    //accumulate products
    wire signed [27:0] n_0_2_sum;
    assign n_0_2_sum = $signed(biases[27-1:18]) + $signed(n_0_2_po_0) + $signed(n_0_2_po_1) + $signed(n_0_2_po_2) + $signed(n_0_2_po_3) + $signed(n_0_2_po_4) + $signed(n_0_2_po_5) + $signed(n_0_2_po_6) + $signed(n_0_2_po_7) + $signed(n_0_2_po_8) + $signed(n_0_2_po_9) + $signed(n_0_2_po_10) + $signed(n_0_2_po_11) + $signed(n_0_2_po_12) + $signed(n_0_2_po_13) + $signed(n_0_2_po_14) + $signed(n_0_2_po_15);
    //relu
    wire [26:0] n_0_2;
    assign n_0_2 = (n_0_2_sum<0) ? $unsigned({27{1'b0}}) : $unsigned(n_0_2_sum);

// layer: 1 - neuron: 0
    //weight 7 : 8'b00000111
    wire signed [22-1:0] n_1_0_po_0;
    assign n_1_0_po_0 = $signed({1'b0, n_0_0}) * $signed(weights[392-1:384]);

    //weight abs(-4) : 8'b00000100
    wire signed [22-1:0] n_1_0_po_1;
    assign n_1_0_po_1 = $signed({1'b0, n_0_1}) * $signed(weights[400-1:392]);

    //weight abs(-19) : 8'b00010011
    wire signed [22-1:0] n_1_0_po_2;
    assign n_1_0_po_2 = $signed({1'b0, n_0_2}) * $signed(weights[408-1:400]);

    //accumulate products
    wire signed [24:0] n_1_0_sum;
    assign n_1_0_sum = $signed(biases[42-1:27]) + $signed(n_1_0_po_0) + $signed(n_1_0_po_1) + $signed(n_1_0_po_2);
    //relu
    wire [23:0] n_1_0;
    assign n_1_0 = (n_1_0_sum<0) ? $unsigned({24{1'b0}}) : $unsigned(n_1_0_sum);

// layer: 1 - neuron: 1
    //weight abs(-32) : 8'b00100000
    wire signed [22-1:0] n_1_1_po_0;
    assign n_1_1_po_0 = $signed({1'b0, n_0_0}) * $signed(weights[416-1:408]);

    //weight abs(-32) : 8'b00100000
    wire signed [22-1:0] n_1_1_po_1;
    assign n_1_1_po_1 = $signed({1'b0, n_0_1}) * $signed(weights[424-1:416]);

    //weight 0 : skip
    //weight abs(0) : 8'b00000000
    wire signed [22-1:0] n_1_1_po_2;
    assign n_1_1_po_2 = $signed({1'b0, n_0_2}) * $signed(weights[432-1:424]);

    //accumulate products
    wire signed [24:0] n_1_1_sum;
    assign n_1_1_sum = $signed(biases[57-1:42]) + $signed(n_1_1_po_0) + $signed(n_1_1_po_1) + $signed(n_1_1_po_2);
    //relu
    wire [23:0] n_1_1;
    assign n_1_1 = (n_1_1_sum<0) ? $unsigned({24{1'b0}}) : $unsigned(n_1_1_sum);

// layer: 1 - neuron: 2
    //weight abs(-36) : 8'b00100100
    wire signed [22-1:0] n_1_2_po_0;
    assign n_1_2_po_0 = $signed({1'b0, n_0_0}) * $signed(weights[440-1:432]);

    //weight abs(-17) : 8'b00010001
    wire signed [22-1:0] n_1_2_po_1;
    assign n_1_2_po_1 = $signed({1'b0, n_0_1}) * $signed(weights[448-1:440]);

    //weight abs(-32) : 8'b00100000
    wire signed [22-1:0] n_1_2_po_2;
    assign n_1_2_po_2 = $signed({1'b0, n_0_2}) * $signed(weights[456-1:448]);

    //accumulate products
    wire signed [24:0] n_1_2_sum;
    assign n_1_2_sum = $signed(biases[72-1:57]) + $signed(n_1_2_po_0) + $signed(n_1_2_po_1) + $signed(n_1_2_po_2);
    //relu
    wire [23:0] n_1_2;
    assign n_1_2 = (n_1_2_sum<0) ? $unsigned({24{1'b0}}) : $unsigned(n_1_2_sum);

// layer: 1 - neuron: 3
    //weight abs(-45) : 8'b00101101
    wire signed [22-1:0] n_1_3_po_0;
    assign n_1_3_po_0 = $signed({1'b0, n_0_0}) * $signed(weights[464-1:456]);

    //weight 10 : 8'b00001010
    wire signed [22-1:0] n_1_3_po_1;
    assign n_1_3_po_1 = $signed({1'b0, n_0_1}) * $signed(weights[472-1:464]);

    //weight 16 : 8'b00010000
    wire signed [22-1:0] n_1_3_po_2;
    assign n_1_3_po_2 = $signed({1'b0, n_0_2}) * $signed(weights[480-1:472]);

    //accumulate products
    wire signed [24:0] n_1_3_sum;
    assign n_1_3_sum = $signed(biases[87-1:72]) + $signed(n_1_3_po_0) + $signed(n_1_3_po_1) + $signed(n_1_3_po_2);
    //relu
    wire [23:0] n_1_3;
    assign n_1_3 = (n_1_3_sum<0) ? $unsigned({24{1'b0}}) : $unsigned(n_1_3_sum);

// layer: 1 - neuron: 4
    //weight 9 : 8'b00001001
    wire signed [22-1:0] n_1_4_po_0;
    assign n_1_4_po_0 = $signed({1'b0, n_0_0}) * $signed(weights[488-1:480]);

    //weight abs(-73) : 8'b01001001
    wire signed [22-1:0] n_1_4_po_1;
    assign n_1_4_po_1 = $signed({1'b0, n_0_1}) * $signed(weights[496-1:488]);

    //weight abs(-7) : 8'b00000111
    wire signed [22-1:0] n_1_4_po_2;
    assign n_1_4_po_2 = $signed({1'b0, n_0_2}) * $signed(weights[504-1:496]);

    //accumulate products
    wire signed [24:0] n_1_4_sum;
    assign n_1_4_sum = $signed(biases[102-1:87]) + $signed(n_1_4_po_0) + $signed(n_1_4_po_1) + $signed(n_1_4_po_2);
    //relu
    wire [23:0] n_1_4;
    assign n_1_4 = (n_1_4_sum<0) ? $unsigned({24{1'b0}}) : $unsigned(n_1_4_sum);

// layer: 1 - neuron: 5
    //weight abs(-2) : 8'b00000010
    wire signed [22-1:0] n_1_5_po_0;
    assign n_1_5_po_0 = $signed({1'b0, n_0_0}) * $signed(weights[512-1:504]);

    //weight 9 : 8'b00001001
    wire signed [22-1:0] n_1_5_po_1;
    assign n_1_5_po_1 = $signed({1'b0, n_0_1}) * $signed(weights[520-1:512]);

    //weight 11 : 8'b00001011
    wire signed [22-1:0] n_1_5_po_2;
    assign n_1_5_po_2 = $signed({1'b0, n_0_2}) * $signed(weights[528-1:520]);

    //accumulate products
    wire signed [24:0] n_1_5_sum;
    assign n_1_5_sum = $signed(biases[117-1:102]) + $signed(n_1_5_po_0) + $signed(n_1_5_po_1) + $signed(n_1_5_po_2);
    //relu
    wire [23:0] n_1_5;
    assign n_1_5 = (n_1_5_sum<0) ? $unsigned({24{1'b0}}) : $unsigned(n_1_5_sum);

// layer: 1 - neuron: 6
    //weight 12 : 8'b00001100
    wire signed [22-1:0] n_1_6_po_0;
    assign n_1_6_po_0 = $signed({1'b0, n_0_0}) * $signed(weights[536-1:528]);

    //weight abs(-17) : 8'b00010001
    wire signed [22-1:0] n_1_6_po_1;
    assign n_1_6_po_1 = $signed({1'b0, n_0_1}) * $signed(weights[544-1:536]);

    //weight abs(-11) : 8'b00001011
    wire signed [22-1:0] n_1_6_po_2;
    assign n_1_6_po_2 = $signed({1'b0, n_0_2}) * $signed(weights[552-1:544]);

    //accumulate products
    wire signed [24:0] n_1_6_sum;
    assign n_1_6_sum = $signed(biases[132-1:117]) + $signed(n_1_6_po_0) + $signed(n_1_6_po_1) + $signed(n_1_6_po_2);
    //relu
    wire [23:0] n_1_6;
    assign n_1_6 = (n_1_6_sum<0) ? $unsigned({24{1'b0}}) : $unsigned(n_1_6_sum);

// layer: 1 - neuron: 7
    //weight abs(-17) : 8'b00010001
    wire signed [22-1:0] n_1_7_po_0;
    assign n_1_7_po_0 = $signed({1'b0, n_0_0}) * $signed(weights[560-1:552]);

    //weight 7 : 8'b00000111
    wire signed [22-1:0] n_1_7_po_1;
    assign n_1_7_po_1 = $signed({1'b0, n_0_1}) * $signed(weights[568-1:560]);

    //weight abs(-4) : 8'b00000100
    wire signed [22-1:0] n_1_7_po_2;
    assign n_1_7_po_2 = $signed({1'b0, n_0_2}) * $signed(weights[576-1:568]);

    //accumulate products
    wire signed [24:0] n_1_7_sum;
    assign n_1_7_sum = $signed(biases[147-1:132]) + $signed(n_1_7_po_0) + $signed(n_1_7_po_1) + $signed(n_1_7_po_2);
    //relu
    wire [23:0] n_1_7;
    assign n_1_7 = (n_1_7_sum<0) ? $unsigned({24{1'b0}}) : $unsigned(n_1_7_sum);

// layer: 1 - neuron: 8
    //weight 0 : skip
    //weight abs(0) : 8'b00000000
    wire signed [22-1:0] n_1_8_po_0;
    assign n_1_8_po_0 = $signed({1'b0, n_0_0}) * $signed(weights[584-1:576]);

    //weight 1 : 8'b00000001
    wire signed [22-1:0] n_1_8_po_1;
    assign n_1_8_po_1 = $signed({1'b0, n_0_1}) * $signed(weights[592-1:584]);

    //weight abs(-30) : 8'b00011110
    wire signed [22-1:0] n_1_8_po_2;
    assign n_1_8_po_2 = $signed({1'b0, n_0_2}) * $signed(weights[600-1:592]);

    //accumulate products
    wire signed [24:0] n_1_8_sum;
    assign n_1_8_sum = $signed(biases[162-1:147]) + $signed(n_1_8_po_0) + $signed(n_1_8_po_1) + $signed(n_1_8_po_2);
    //relu
    wire [23:0] n_1_8;
    assign n_1_8 = (n_1_8_sum<0) ? $unsigned({24{1'b0}}) : $unsigned(n_1_8_sum);

// layer: 1 - neuron: 9
    //weight abs(-2) : 8'b00000010
    wire signed [22-1:0] n_1_9_po_0;
    assign n_1_9_po_0 = $signed({1'b0, n_0_0}) * $signed(weights[608-1:600]);

    //weight abs(-5) : 8'b00000101
    wire signed [22-1:0] n_1_9_po_1;
    assign n_1_9_po_1 = $signed({1'b0, n_0_1}) * $signed(weights[616-1:608]);

    //weight 9 : 8'b00001001
    wire signed [22-1:0] n_1_9_po_2;
    assign n_1_9_po_2 = $signed({1'b0, n_0_2}) * $signed(weights[624-1:616]);

    //accumulate products
    wire signed [24:0] n_1_9_sum;
    assign n_1_9_sum = $signed(biases[177-1:162]) + $signed(n_1_9_po_0) + $signed(n_1_9_po_1) + $signed(n_1_9_po_2);
    //relu
    wire [23:0] n_1_9;
    assign n_1_9 = (n_1_9_sum<0) ? $unsigned({24{1'b0}}) : $unsigned(n_1_9_sum);

// argmax: 10 classes, need 4 bits
// argmax inp: n_1_0, n_1_1, n_1_2, n_1_3, n_1_4, n_1_5, n_1_6, n_1_7, n_1_8, n_1_9
    //comp level 0
    wire cmp_0_0;
    wire [26:0] argmax_val_0_0;
    wire [3:0] argmax_idx_0_0;
    assign {cmp_0_0} = ( n_1_0 >= n_1_1 );
    assign {argmax_val_0_0} = ( cmp_0_0 ) ? n_1_0 : n_1_1;
    assign {argmax_idx_0_0} = ( cmp_0_0 ) ? 4'b0000 : 4'b0001;

    wire cmp_0_2;
    wire [26:0] argmax_val_0_2;
    wire [3:0] argmax_idx_0_2;
    assign {cmp_0_2} = ( n_1_2 >= n_1_3 );
    assign {argmax_val_0_2} = ( cmp_0_2 ) ? n_1_2 : n_1_3;
    assign {argmax_idx_0_2} = ( cmp_0_2 ) ? 4'b0010 : 4'b0011;

    wire cmp_0_4;
    wire [26:0] argmax_val_0_4;
    wire [3:0] argmax_idx_0_4;
    assign {cmp_0_4} = ( n_1_4 >= n_1_5 );
    assign {argmax_val_0_4} = ( cmp_0_4 ) ? n_1_4 : n_1_5;
    assign {argmax_idx_0_4} = ( cmp_0_4 ) ? 4'b0100 : 4'b0101;

    wire cmp_0_6;
    wire [26:0] argmax_val_0_6;
    wire [3:0] argmax_idx_0_6;
    assign {cmp_0_6} = ( n_1_6 >= n_1_7 );
    assign {argmax_val_0_6} = ( cmp_0_6 ) ? n_1_6 : n_1_7;
    assign {argmax_idx_0_6} = ( cmp_0_6 ) ? 4'b0110 : 4'b0111;

    wire cmp_0_8;
    wire [26:0] argmax_val_0_8;
    wire [3:0] argmax_idx_0_8;
    assign {cmp_0_8} = ( n_1_8 >= n_1_9 );
    assign {argmax_val_0_8} = ( cmp_0_8 ) ? n_1_8 : n_1_9;
    assign {argmax_idx_0_8} = ( cmp_0_8 ) ? 4'b1000 : 4'b1001;

    //comp level 1
    wire cmp_1_0;
    wire [26:0] argmax_val_1_0;
    wire [3:0] argmax_idx_1_0;
    assign {cmp_1_0} = ( argmax_val_0_0 >= argmax_val_0_2 );
    assign {argmax_val_1_0} = ( cmp_1_0 ) ? argmax_val_0_0 : argmax_val_0_2;
    assign {argmax_idx_1_0} = ( cmp_1_0 ) ? argmax_idx_0_0 : argmax_idx_0_2;

    wire cmp_1_2;
    wire [26:0] argmax_val_1_2;
    wire [3:0] argmax_idx_1_2;
    assign {cmp_1_2} = ( argmax_val_0_4 >= argmax_val_0_6 );
    assign {argmax_val_1_2} = ( cmp_1_2 ) ? argmax_val_0_4 : argmax_val_0_6;
    assign {argmax_idx_1_2} = ( cmp_1_2 ) ? argmax_idx_0_4 : argmax_idx_0_6;

    //comp level 2
    wire cmp_2_0;
    wire [26:0] argmax_val_2_0;
    wire [3:0] argmax_idx_2_0;
    assign {cmp_2_0} = ( argmax_val_1_0 >= argmax_val_1_2 );
    assign {argmax_val_2_0} = ( cmp_2_0 ) ? argmax_val_1_0 : argmax_val_1_2;
    assign {argmax_idx_2_0} = ( cmp_2_0 ) ? argmax_idx_1_0 : argmax_idx_1_2;

    //comp level 3
    wire cmp_3_0;
    wire [26:0] argmax_val_3_0;
    wire [3:0] argmax_idx_3_0;
    assign {cmp_3_0} = ( argmax_val_2_0 >= argmax_val_0_8 );
    assign {argmax_val_3_0} = ( cmp_3_0 ) ? argmax_val_2_0 : argmax_val_0_8;
    assign {argmax_idx_3_0} = ( cmp_3_0 ) ? argmax_idx_2_0 : argmax_idx_0_8;

    assign out = argmax_idx_3_0;

endmodule

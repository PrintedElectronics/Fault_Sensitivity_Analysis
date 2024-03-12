//generic design, we don't actually know the weights and intercepts, they are given to the network through the testbench.
//weights: [[[-93, -27, -55, -80, 84, -10, 18, 15], [-18, -22, 5, -14, 80, -24, 64, 43], [2, 16, -36, 58, -43, -19, -66, 8]], [[4, -25, 42], [68, -23, -25], [-102, 37, -46]]]
//intercepts: [[473, -912, 407], [-186, 669, -3346]]
module top (inp, weights, biases, out);
input [31:0] inp;
input [263:0] weights;
input [71:0] biases;
output [1:0] out;

// layer: 0 - neuron: 0
    //weight abs(-93) : 8'b01011101
    wire signed [12-1:0] n_0_0_po_0;
    assign n_0_0_po_0 = $signed({1'b0, inp[3:0]}) * $signed(weights[8-1:0]);

    //weight abs(-27) : 8'b00011011
    wire signed [12-1:0] n_0_0_po_1;
    assign n_0_0_po_1 = $signed({1'b0, inp[7:4]}) * $signed(weights[16-1:8]);

    //weight abs(-55) : 8'b00110111
    wire signed [12-1:0] n_0_0_po_2;
    assign n_0_0_po_2 = $signed({1'b0, inp[11:8]}) * $signed(weights[24-1:16]);

    //weight abs(-80) : 8'b01010000
    wire signed [12-1:0] n_0_0_po_3;
    assign n_0_0_po_3 = $signed({1'b0, inp[15:12]}) * $signed(weights[32-1:24]);

    //weight 84 : 8'b01010100
    wire signed [12-1:0] n_0_0_po_4;
    assign n_0_0_po_4 = $signed({1'b0, inp[19:16]}) * $signed(weights[40-1:32]);

    //weight abs(-10) : 8'b00001010
    wire signed [12-1:0] n_0_0_po_5;
    assign n_0_0_po_5 = $signed({1'b0, inp[23:20]}) * $signed(weights[48-1:40]);

    //weight 18 : 8'b00010010
    wire signed [12-1:0] n_0_0_po_6;
    assign n_0_0_po_6 = $signed({1'b0, inp[27:24]}) * $signed(weights[56-1:48]);

    //weight 15 : 8'b00001111
    wire signed [12-1:0] n_0_0_po_7;
    assign n_0_0_po_7 = $signed({1'b0, inp[31:28]}) * $signed(weights[64-1:56]);

    //accumulate products
    wire signed [19:0] n_0_0_sum;
    assign n_0_0_sum = $signed(biases[11-1:0]) + $signed(n_0_0_po_0) + $signed(n_0_0_po_1) + $signed(n_0_0_po_2) + $signed(n_0_0_po_3) + $signed(n_0_0_po_4) + $signed(n_0_0_po_5) + $signed(n_0_0_po_6) + $signed(n_0_0_po_7);
    //relu
    wire [18:0] n_0_0;
    assign n_0_0 = (n_0_0_sum<0) ? $unsigned({19{1'b0}}) : $unsigned(n_0_0_sum);

// layer: 0 - neuron: 1
    //weight abs(-18) : 8'b00010010
    wire signed [12-1:0] n_0_1_po_0;
    assign n_0_1_po_0 = $signed({1'b0, inp[3:0]}) * $signed(weights[72-1:64]);

    //weight abs(-22) : 8'b00010110
    wire signed [12-1:0] n_0_1_po_1;
    assign n_0_1_po_1 = $signed({1'b0, inp[7:4]}) * $signed(weights[80-1:72]);

    //weight 5 : 8'b00000101
    wire signed [12-1:0] n_0_1_po_2;
    assign n_0_1_po_2 = $signed({1'b0, inp[11:8]}) * $signed(weights[88-1:80]);

    //weight abs(-14) : 8'b00001110
    wire signed [12-1:0] n_0_1_po_3;
    assign n_0_1_po_3 = $signed({1'b0, inp[15:12]}) * $signed(weights[96-1:88]);

    //weight 80 : 8'b01010000
    wire signed [12-1:0] n_0_1_po_4;
    assign n_0_1_po_4 = $signed({1'b0, inp[19:16]}) * $signed(weights[104-1:96]);

    //weight abs(-24) : 8'b00011000
    wire signed [12-1:0] n_0_1_po_5;
    assign n_0_1_po_5 = $signed({1'b0, inp[23:20]}) * $signed(weights[112-1:104]);

    //weight 64 : 8'b01000000
    wire signed [12-1:0] n_0_1_po_6;
    assign n_0_1_po_6 = $signed({1'b0, inp[27:24]}) * $signed(weights[120-1:112]);

    //weight 43 : 8'b00101011
    wire signed [12-1:0] n_0_1_po_7;
    assign n_0_1_po_7 = $signed({1'b0, inp[31:28]}) * $signed(weights[128-1:120]);

    //accumulate products
    wire signed [19:0] n_0_1_sum;
    assign n_0_1_sum = $signed(biases[22-1:11]) + $signed(n_0_1_po_0) + $signed(n_0_1_po_1) + $signed(n_0_1_po_2) + $signed(n_0_1_po_3) + $signed(n_0_1_po_4) + $signed(n_0_1_po_5) + $signed(n_0_1_po_6) + $signed(n_0_1_po_7);
    //relu
    wire [18:0] n_0_1;
    assign n_0_1 = (n_0_1_sum<0) ? $unsigned({19{1'b0}}) : $unsigned(n_0_1_sum);

// layer: 0 - neuron: 2
    //weight 2 : 8'b00000010
    wire signed [12-1:0] n_0_2_po_0;
    assign n_0_2_po_0 = $signed({1'b0, inp[3:0]}) * $signed(weights[136-1:128]);

    //weight 16 : 8'b00010000
    wire signed [12-1:0] n_0_2_po_1;
    assign n_0_2_po_1 = $signed({1'b0, inp[7:4]}) * $signed(weights[144-1:136]);

    //weight abs(-36) : 8'b00100100
    wire signed [12-1:0] n_0_2_po_2;
    assign n_0_2_po_2 = $signed({1'b0, inp[11:8]}) * $signed(weights[152-1:144]);

    //weight 58 : 8'b00111010
    wire signed [12-1:0] n_0_2_po_3;
    assign n_0_2_po_3 = $signed({1'b0, inp[15:12]}) * $signed(weights[160-1:152]);

    //weight abs(-43) : 8'b00101011
    wire signed [12-1:0] n_0_2_po_4;
    assign n_0_2_po_4 = $signed({1'b0, inp[19:16]}) * $signed(weights[168-1:160]);

    //weight abs(-19) : 8'b00010011
    wire signed [12-1:0] n_0_2_po_5;
    assign n_0_2_po_5 = $signed({1'b0, inp[23:20]}) * $signed(weights[176-1:168]);

    //weight abs(-66) : 8'b01000010
    wire signed [12-1:0] n_0_2_po_6;
    assign n_0_2_po_6 = $signed({1'b0, inp[27:24]}) * $signed(weights[184-1:176]);

    //weight 8 : 8'b00001000
    wire signed [12-1:0] n_0_2_po_7;
    assign n_0_2_po_7 = $signed({1'b0, inp[31:28]}) * $signed(weights[192-1:184]);

    //accumulate products
    wire signed [19:0] n_0_2_sum;
    assign n_0_2_sum = $signed(biases[33-1:22]) + $signed(n_0_2_po_0) + $signed(n_0_2_po_1) + $signed(n_0_2_po_2) + $signed(n_0_2_po_3) + $signed(n_0_2_po_4) + $signed(n_0_2_po_5) + $signed(n_0_2_po_6) + $signed(n_0_2_po_7);
    //relu
    wire [18:0] n_0_2;
    assign n_0_2 = (n_0_2_sum<0) ? $unsigned({19{1'b0}}) : $unsigned(n_0_2_sum);

// layer: 1 - neuron: 0
    //weight 4 : 8'b00000100
    wire signed [22-1:0] n_1_0_po_0;
    assign n_1_0_po_0 = $signed({1'b0, n_0_0}) * $signed(weights[200-1:192]);

    //weight abs(-25) : 8'b00011001
    wire signed [22-1:0] n_1_0_po_1;
    assign n_1_0_po_1 = $signed({1'b0, n_0_1}) * $signed(weights[208-1:200]);

    //weight 42 : 8'b00101010
    wire signed [22-1:0] n_1_0_po_2;
    assign n_1_0_po_2 = $signed({1'b0, n_0_2}) * $signed(weights[216-1:208]);

    //accumulate products
    wire signed [24:0] n_1_0_sum;
    assign n_1_0_sum = $signed(biases[46-1:33]) + $signed(n_1_0_po_0) + $signed(n_1_0_po_1) + $signed(n_1_0_po_2);
    //relu
    wire [23:0] n_1_0;
    assign n_1_0 = (n_1_0_sum<0) ? $unsigned({24{1'b0}}) : $unsigned(n_1_0_sum);

// layer: 1 - neuron: 1
    //weight 68 : 8'b01000100
    wire signed [22-1:0] n_1_1_po_0;
    assign n_1_1_po_0 = $signed({1'b0, n_0_0}) * $signed(weights[224-1:216]);

    //weight abs(-23) : 8'b00010111
    wire signed [22-1:0] n_1_1_po_1;
    assign n_1_1_po_1 = $signed({1'b0, n_0_1}) * $signed(weights[232-1:224]);

    //weight abs(-25) : 8'b00011001
    wire signed [22-1:0] n_1_1_po_2;
    assign n_1_1_po_2 = $signed({1'b0, n_0_2}) * $signed(weights[240-1:232]);

    //accumulate products
    wire signed [24:0] n_1_1_sum;
    assign n_1_1_sum = $signed(biases[59-1:46]) + $signed(n_1_1_po_0) + $signed(n_1_1_po_1) + $signed(n_1_1_po_2);
    //relu
    wire [23:0] n_1_1;
    assign n_1_1 = (n_1_1_sum<0) ? $unsigned({24{1'b0}}) : $unsigned(n_1_1_sum);

// layer: 1 - neuron: 2
    //weight abs(-102) : 8'b01100110
    wire signed [22-1:0] n_1_2_po_0;
    assign n_1_2_po_0 = $signed({1'b0, n_0_0}) * $signed(weights[248-1:240]);

    //weight 37 : 8'b00100101
    wire signed [22-1:0] n_1_2_po_1;
    assign n_1_2_po_1 = $signed({1'b0, n_0_1}) * $signed(weights[256-1:248]);

    //weight abs(-46) : 8'b00101110
    wire signed [22-1:0] n_1_2_po_2;
    assign n_1_2_po_2 = $signed({1'b0, n_0_2}) * $signed(weights[264-1:256]);

    //accumulate products
    wire signed [24:0] n_1_2_sum;
    assign n_1_2_sum = $signed(biases[72-1:59]) + $signed(n_1_2_po_0) + $signed(n_1_2_po_1) + $signed(n_1_2_po_2);
    //relu
    wire [23:0] n_1_2;
    assign n_1_2 = (n_1_2_sum<0) ? $unsigned({24{1'b0}}) : $unsigned(n_1_2_sum);

// argmax: 3 classes, need 2 bits
// argmax inp: n_1_0, n_1_1, n_1_2
    //comp level 0
    wire cmp_0_0;
    wire [23:0] argmax_val_0_0;
    wire [1:0] argmax_idx_0_0;
    assign {cmp_0_0} = ( n_1_0 >= n_1_1 );
    assign {argmax_val_0_0} = ( cmp_0_0 ) ? n_1_0 : n_1_1;
    assign {argmax_idx_0_0} = ( cmp_0_0 ) ? 2'b00 : 2'b01;

    //comp level 1
    wire cmp_1_0;
    wire [23:0] argmax_val_1_0;
    wire [1:0] argmax_idx_1_0;
    assign {cmp_1_0} = ( argmax_val_0_0 >= n_1_2 );
    assign {argmax_val_1_0} = ( cmp_1_0 ) ? argmax_val_0_0 : n_1_2;
    assign {argmax_idx_1_0} = ( cmp_1_0 ) ? argmax_idx_0_0 : 2'b10;

    assign out = argmax_idx_1_0;

endmodule

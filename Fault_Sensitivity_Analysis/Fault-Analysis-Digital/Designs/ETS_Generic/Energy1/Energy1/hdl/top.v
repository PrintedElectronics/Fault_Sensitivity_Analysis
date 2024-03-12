//generic design, we don't actually know the weights and intercepts, they are given to the network through the testbench.
//weights: [[[-6, -1, -6, -2, -5, -5, -4, -4], [37, -18, 31, 44, 75, -1, 6, -1], [-77, 1, -85, -78, 104, 0, -18, 4]], [[4, -19, 95], [-4, 0, 7], [-3, 5, -7]]]
//intercepts: [[-148, -576, 145], [5138, -579, -5864]]
module top (inp, weights, biases, out);
input [31:0] inp;
input [263:0] weights;
input [74:0] biases;
output [1:0] out;

// layer: 0 - neuron: 0
    //weight abs(-6) : 8'b00000110
    wire signed [12-1:0] n_0_0_po_0;
    assign n_0_0_po_0 = $signed({1'b0, inp[3:0]}) * $signed(weights[8-1:0]);

    //weight abs(-1) : 8'b00000001
    wire signed [12-1:0] n_0_0_po_1;
    assign n_0_0_po_1 = $signed({1'b0, inp[7:4]}) * $signed(weights[16-1:8]);

    //weight abs(-6) : 8'b00000110
    wire signed [12-1:0] n_0_0_po_2;
    assign n_0_0_po_2 = $signed({1'b0, inp[11:8]}) * $signed(weights[24-1:16]);

    //weight abs(-2) : 8'b00000010
    wire signed [12-1:0] n_0_0_po_3;
    assign n_0_0_po_3 = $signed({1'b0, inp[15:12]}) * $signed(weights[32-1:24]);

    //weight abs(-5) : 8'b00000101
    wire signed [12-1:0] n_0_0_po_4;
    assign n_0_0_po_4 = $signed({1'b0, inp[19:16]}) * $signed(weights[40-1:32]);

    //weight abs(-5) : 8'b00000101
    wire signed [12-1:0] n_0_0_po_5;
    assign n_0_0_po_5 = $signed({1'b0, inp[23:20]}) * $signed(weights[48-1:40]);

    //weight abs(-4) : 8'b00000100
    wire signed [12-1:0] n_0_0_po_6;
    assign n_0_0_po_6 = $signed({1'b0, inp[27:24]}) * $signed(weights[56-1:48]);

    //weight abs(-4) : 8'b00000100
    wire signed [12-1:0] n_0_0_po_7;
    assign n_0_0_po_7 = $signed({1'b0, inp[31:28]}) * $signed(weights[64-1:56]);

    //accumulate products
    wire signed [19:0] n_0_0_sum;
    assign n_0_0_sum = $signed(biases[11-1:0]) + $signed(n_0_0_po_0) + $signed(n_0_0_po_1) + $signed(n_0_0_po_2) + $signed(n_0_0_po_3) + $signed(n_0_0_po_4) + $signed(n_0_0_po_5) + $signed(n_0_0_po_6) + $signed(n_0_0_po_7);
    //relu
    wire [18:0] n_0_0;
    assign n_0_0 = (n_0_0_sum<0) ? $unsigned({19{1'b0}}) : $unsigned(n_0_0_sum);

// layer: 0 - neuron: 1
    //weight 37 : 8'b00100101
    wire signed [12-1:0] n_0_1_po_0;
    assign n_0_1_po_0 = $signed({1'b0, inp[3:0]}) * $signed(weights[72-1:64]);

    //weight abs(-18) : 8'b00010010
    wire signed [12-1:0] n_0_1_po_1;
    assign n_0_1_po_1 = $signed({1'b0, inp[7:4]}) * $signed(weights[80-1:72]);

    //weight 31 : 8'b00011111
    wire signed [12-1:0] n_0_1_po_2;
    assign n_0_1_po_2 = $signed({1'b0, inp[11:8]}) * $signed(weights[88-1:80]);

    //weight 44 : 8'b00101100
    wire signed [12-1:0] n_0_1_po_3;
    assign n_0_1_po_3 = $signed({1'b0, inp[15:12]}) * $signed(weights[96-1:88]);

    //weight 75 : 8'b01001011
    wire signed [12-1:0] n_0_1_po_4;
    assign n_0_1_po_4 = $signed({1'b0, inp[19:16]}) * $signed(weights[104-1:96]);

    //weight abs(-1) : 8'b00000001
    wire signed [12-1:0] n_0_1_po_5;
    assign n_0_1_po_5 = $signed({1'b0, inp[23:20]}) * $signed(weights[112-1:104]);

    //weight 6 : 8'b00000110
    wire signed [12-1:0] n_0_1_po_6;
    assign n_0_1_po_6 = $signed({1'b0, inp[27:24]}) * $signed(weights[120-1:112]);

    //weight abs(-1) : 8'b00000001
    wire signed [12-1:0] n_0_1_po_7;
    assign n_0_1_po_7 = $signed({1'b0, inp[31:28]}) * $signed(weights[128-1:120]);

    //accumulate products
    wire signed [19:0] n_0_1_sum;
    assign n_0_1_sum = $signed(biases[22-1:11]) + $signed(n_0_1_po_0) + $signed(n_0_1_po_1) + $signed(n_0_1_po_2) + $signed(n_0_1_po_3) + $signed(n_0_1_po_4) + $signed(n_0_1_po_5) + $signed(n_0_1_po_6) + $signed(n_0_1_po_7);
    //relu
    wire [18:0] n_0_1;
    assign n_0_1 = (n_0_1_sum<0) ? $unsigned({19{1'b0}}) : $unsigned(n_0_1_sum);

// layer: 0 - neuron: 2
    //weight abs(-77) : 8'b01001101
    wire signed [12-1:0] n_0_2_po_0;
    assign n_0_2_po_0 = $signed({1'b0, inp[3:0]}) * $signed(weights[136-1:128]);

    //weight 1 : 8'b00000001
    wire signed [12-1:0] n_0_2_po_1;
    assign n_0_2_po_1 = $signed({1'b0, inp[7:4]}) * $signed(weights[144-1:136]);

    //weight abs(-85) : 8'b01010101
    wire signed [12-1:0] n_0_2_po_2;
    assign n_0_2_po_2 = $signed({1'b0, inp[11:8]}) * $signed(weights[152-1:144]);

    //weight abs(-78) : 8'b01001110
    wire signed [12-1:0] n_0_2_po_3;
    assign n_0_2_po_3 = $signed({1'b0, inp[15:12]}) * $signed(weights[160-1:152]);

    //weight 104 : 8'b01101000
    wire signed [12-1:0] n_0_2_po_4;
    assign n_0_2_po_4 = $signed({1'b0, inp[19:16]}) * $signed(weights[168-1:160]);

    //weight 0 : skip
    //weight abs(0) : 8'b00000000
    wire signed [12-1:0] n_0_2_po_5;
    assign n_0_2_po_5 = $signed({1'b0, inp[23:20]}) * $signed(weights[176-1:168]);

    //weight abs(-18) : 8'b00010010
    wire signed [12-1:0] n_0_2_po_6;
    assign n_0_2_po_6 = $signed({1'b0, inp[27:24]}) * $signed(weights[184-1:176]);

    //weight 4 : 8'b00000100
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

    //weight abs(-19) : 8'b00010011
    wire signed [22-1:0] n_1_0_po_1;
    assign n_1_0_po_1 = $signed({1'b0, n_0_1}) * $signed(weights[208-1:200]);

    //weight 95 : 8'b01011111
    wire signed [22-1:0] n_1_0_po_2;
    assign n_1_0_po_2 = $signed({1'b0, n_0_2}) * $signed(weights[216-1:208]);

    //accumulate products
    wire signed [24:0] n_1_0_sum;
    assign n_1_0_sum = $signed(biases[47-1:33]) + $signed(n_1_0_po_0) + $signed(n_1_0_po_1) + $signed(n_1_0_po_2);
    //relu
    wire [23:0] n_1_0;
    assign n_1_0 = (n_1_0_sum<0) ? $unsigned({24{1'b0}}) : $unsigned(n_1_0_sum);

// layer: 1 - neuron: 1
    //weight abs(-4) : 8'b00000100
    wire signed [22-1:0] n_1_1_po_0;
    assign n_1_1_po_0 = $signed({1'b0, n_0_0}) * $signed(weights[224-1:216]);

    //weight 0 : skip
    //weight abs(0) : 8'b00000000
    wire signed [22-1:0] n_1_1_po_1;
    assign n_1_1_po_1 = $signed({1'b0, n_0_1}) * $signed(weights[232-1:224]);

    //weight 7 : 8'b00000111
    wire signed [22-1:0] n_1_1_po_2;
    assign n_1_1_po_2 = $signed({1'b0, n_0_2}) * $signed(weights[240-1:232]);

    //accumulate products
    wire signed [24:0] n_1_1_sum;
    assign n_1_1_sum = $signed(biases[61-1:47]) + $signed(n_1_1_po_0) + $signed(n_1_1_po_1) + $signed(n_1_1_po_2);
    //relu
    wire [23:0] n_1_1;
    assign n_1_1 = (n_1_1_sum<0) ? $unsigned({24{1'b0}}) : $unsigned(n_1_1_sum);

// layer: 1 - neuron: 2
    //weight abs(-3) : 8'b00000011
    wire signed [22-1:0] n_1_2_po_0;
    assign n_1_2_po_0 = $signed({1'b0, n_0_0}) * $signed(weights[248-1:240]);

    //weight 5 : 8'b00000101
    wire signed [22-1:0] n_1_2_po_1;
    assign n_1_2_po_1 = $signed({1'b0, n_0_1}) * $signed(weights[256-1:248]);

    //weight abs(-7) : 8'b00000111
    wire signed [22-1:0] n_1_2_po_2;
    assign n_1_2_po_2 = $signed({1'b0, n_0_2}) * $signed(weights[264-1:256]);

    //accumulate products
    wire signed [24:0] n_1_2_sum;
    assign n_1_2_sum = $signed(biases[75-1:61]) + $signed(n_1_2_po_0) + $signed(n_1_2_po_1) + $signed(n_1_2_po_2);
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

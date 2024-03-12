//generic design, we don't actually know the weights and intercepts, they are given to the network through the testbench.
//weights: [[[-13, 22, -8, -17, -30, 38], [-2, -2, 2, 1, -2, -3], [-3, -1, -1, -3, -15, 88]], [[7, 0, -31], [-7, 3, -50], [-2, 1, 51]]]
//intercepts: [[331, -51, -71], [-903, 750, -759]]
module top (inp, weights, biases, out);
input [23:0] inp;
input [215:0] weights;
input [62:0] biases;
output [1:0] out;

// layer: 0 - neuron: 0
    //weight abs(-13) : 8'b00001101
    wire signed [12-1:0] n_0_0_po_0;
    assign n_0_0_po_0 = $signed({1'b0, inp[3:0]}) * $signed(weights[8-1:0]);

    //weight 22 : 8'b00010110
    wire signed [12-1:0] n_0_0_po_1;
    assign n_0_0_po_1 = $signed({1'b0, inp[7:4]}) * $signed(weights[16-1:8]);

    //weight abs(-8) : 8'b00001000
    wire signed [12-1:0] n_0_0_po_2;
    assign n_0_0_po_2 = $signed({1'b0, inp[11:8]}) * $signed(weights[24-1:16]);

    //weight abs(-17) : 8'b00010001
    wire signed [12-1:0] n_0_0_po_3;
    assign n_0_0_po_3 = $signed({1'b0, inp[15:12]}) * $signed(weights[32-1:24]);

    //weight abs(-30) : 8'b00011110
    wire signed [12-1:0] n_0_0_po_4;
    assign n_0_0_po_4 = $signed({1'b0, inp[19:16]}) * $signed(weights[40-1:32]);

    //weight 38 : 8'b00100110
    wire signed [12-1:0] n_0_0_po_5;
    assign n_0_0_po_5 = $signed({1'b0, inp[23:20]}) * $signed(weights[48-1:40]);

    //accumulate products
    wire signed [17:0] n_0_0_sum;
    assign n_0_0_sum = $signed(biases[10-1:0]) + $signed(n_0_0_po_0) + $signed(n_0_0_po_1) + $signed(n_0_0_po_2) + $signed(n_0_0_po_3) + $signed(n_0_0_po_4) + $signed(n_0_0_po_5);
    //relu
    wire [16:0] n_0_0;
    assign n_0_0 = (n_0_0_sum<0) ? $unsigned({17{1'b0}}) : $unsigned(n_0_0_sum);

// layer: 0 - neuron: 1
    //weight abs(-2) : 8'b00000010
    wire signed [12-1:0] n_0_1_po_0;
    assign n_0_1_po_0 = $signed({1'b0, inp[3:0]}) * $signed(weights[56-1:48]);

    //weight abs(-2) : 8'b00000010
    wire signed [12-1:0] n_0_1_po_1;
    assign n_0_1_po_1 = $signed({1'b0, inp[7:4]}) * $signed(weights[64-1:56]);

    //weight 2 : 8'b00000010
    wire signed [12-1:0] n_0_1_po_2;
    assign n_0_1_po_2 = $signed({1'b0, inp[11:8]}) * $signed(weights[72-1:64]);

    //weight 1 : 8'b00000001
    wire signed [12-1:0] n_0_1_po_3;
    assign n_0_1_po_3 = $signed({1'b0, inp[15:12]}) * $signed(weights[80-1:72]);

    //weight abs(-2) : 8'b00000010
    wire signed [12-1:0] n_0_1_po_4;
    assign n_0_1_po_4 = $signed({1'b0, inp[19:16]}) * $signed(weights[88-1:80]);

    //weight abs(-3) : 8'b00000011
    wire signed [12-1:0] n_0_1_po_5;
    assign n_0_1_po_5 = $signed({1'b0, inp[23:20]}) * $signed(weights[96-1:88]);

    //accumulate products
    wire signed [17:0] n_0_1_sum;
    assign n_0_1_sum = $signed(biases[20-1:10]) + $signed(n_0_1_po_0) + $signed(n_0_1_po_1) + $signed(n_0_1_po_2) + $signed(n_0_1_po_3) + $signed(n_0_1_po_4) + $signed(n_0_1_po_5);
    //relu
    wire [16:0] n_0_1;
    assign n_0_1 = (n_0_1_sum<0) ? $unsigned({17{1'b0}}) : $unsigned(n_0_1_sum);

// layer: 0 - neuron: 2
    //weight abs(-3) : 8'b00000011
    wire signed [12-1:0] n_0_2_po_0;
    assign n_0_2_po_0 = $signed({1'b0, inp[3:0]}) * $signed(weights[104-1:96]);

    //weight abs(-1) : 8'b00000001
    wire signed [12-1:0] n_0_2_po_1;
    assign n_0_2_po_1 = $signed({1'b0, inp[7:4]}) * $signed(weights[112-1:104]);

    //weight abs(-1) : 8'b00000001
    wire signed [12-1:0] n_0_2_po_2;
    assign n_0_2_po_2 = $signed({1'b0, inp[11:8]}) * $signed(weights[120-1:112]);

    //weight abs(-3) : 8'b00000011
    wire signed [12-1:0] n_0_2_po_3;
    assign n_0_2_po_3 = $signed({1'b0, inp[15:12]}) * $signed(weights[128-1:120]);

    //weight abs(-15) : 8'b00001111
    wire signed [12-1:0] n_0_2_po_4;
    assign n_0_2_po_4 = $signed({1'b0, inp[19:16]}) * $signed(weights[136-1:128]);

    //weight 88 : 8'b01011000
    wire signed [12-1:0] n_0_2_po_5;
    assign n_0_2_po_5 = $signed({1'b0, inp[23:20]}) * $signed(weights[144-1:136]);

    //accumulate products
    wire signed [17:0] n_0_2_sum;
    assign n_0_2_sum = $signed(biases[30-1:20]) + $signed(n_0_2_po_0) + $signed(n_0_2_po_1) + $signed(n_0_2_po_2) + $signed(n_0_2_po_3) + $signed(n_0_2_po_4) + $signed(n_0_2_po_5);
    //relu
    wire [16:0] n_0_2;
    assign n_0_2 = (n_0_2_sum<0) ? $unsigned({17{1'b0}}) : $unsigned(n_0_2_sum);

// layer: 1 - neuron: 0
    //weight 7 : 8'b00000111
    wire signed [20-1:0] n_1_0_po_0;
    assign n_1_0_po_0 = $signed({1'b0, n_0_0}) * $signed(weights[152-1:144]);

    //weight 0 : skip
    //weight abs(0) : 8'b00000000
    wire signed [20-1:0] n_1_0_po_1;
    assign n_1_0_po_1 = $signed({1'b0, n_0_1}) * $signed(weights[160-1:152]);

    //weight abs(-31) : 8'b00011111
    wire signed [20-1:0] n_1_0_po_2;
    assign n_1_0_po_2 = $signed({1'b0, n_0_2}) * $signed(weights[168-1:160]);

    //accumulate products
    wire signed [22:0] n_1_0_sum;
    assign n_1_0_sum = $signed(biases[41-1:30]) + $signed(n_1_0_po_0) + $signed(n_1_0_po_1) + $signed(n_1_0_po_2);
    //relu
    wire [21:0] n_1_0;
    assign n_1_0 = (n_1_0_sum<0) ? $unsigned({22{1'b0}}) : $unsigned(n_1_0_sum);

// layer: 1 - neuron: 1
    //weight abs(-7) : 8'b00000111
    wire signed [20-1:0] n_1_1_po_0;
    assign n_1_1_po_0 = $signed({1'b0, n_0_0}) * $signed(weights[176-1:168]);

    //weight 3 : 8'b00000011
    wire signed [20-1:0] n_1_1_po_1;
    assign n_1_1_po_1 = $signed({1'b0, n_0_1}) * $signed(weights[184-1:176]);

    //weight abs(-50) : 8'b00110010
    wire signed [20-1:0] n_1_1_po_2;
    assign n_1_1_po_2 = $signed({1'b0, n_0_2}) * $signed(weights[192-1:184]);

    //accumulate products
    wire signed [22:0] n_1_1_sum;
    assign n_1_1_sum = $signed(biases[52-1:41]) + $signed(n_1_1_po_0) + $signed(n_1_1_po_1) + $signed(n_1_1_po_2);
    //relu
    wire [21:0] n_1_1;
    assign n_1_1 = (n_1_1_sum<0) ? $unsigned({22{1'b0}}) : $unsigned(n_1_1_sum);

// layer: 1 - neuron: 2
    //weight abs(-2) : 8'b00000010
    wire signed [20-1:0] n_1_2_po_0;
    assign n_1_2_po_0 = $signed({1'b0, n_0_0}) * $signed(weights[200-1:192]);

    //weight 1 : 8'b00000001
    wire signed [20-1:0] n_1_2_po_1;
    assign n_1_2_po_1 = $signed({1'b0, n_0_1}) * $signed(weights[208-1:200]);

    //weight 51 : 8'b00110011
    wire signed [20-1:0] n_1_2_po_2;
    assign n_1_2_po_2 = $signed({1'b0, n_0_2}) * $signed(weights[216-1:208]);

    //accumulate products
    wire signed [22:0] n_1_2_sum;
    assign n_1_2_sum = $signed(biases[63-1:52]) + $signed(n_1_2_po_0) + $signed(n_1_2_po_1) + $signed(n_1_2_po_2);
    //relu
    wire [21:0] n_1_2;
    assign n_1_2 = (n_1_2_sum<0) ? $unsigned({22{1'b0}}) : $unsigned(n_1_2_sum);

// argmax: 3 classes, need 2 bits
// argmax inp: n_1_0, n_1_1, n_1_2
    //comp level 0
    wire cmp_0_0;
    wire [21:0] argmax_val_0_0;
    wire [1:0] argmax_idx_0_0;
    assign {cmp_0_0} = ( n_1_0 >= n_1_1 );
    assign {argmax_val_0_0} = ( cmp_0_0 ) ? n_1_0 : n_1_1;
    assign {argmax_idx_0_0} = ( cmp_0_0 ) ? 2'b00 : 2'b01;

    //comp level 1
    wire cmp_1_0;
    wire [21:0] argmax_val_1_0;
    wire [1:0] argmax_idx_1_0;
    assign {cmp_1_0} = ( argmax_val_0_0 >= n_1_2 );
    assign {argmax_val_1_0} = ( cmp_1_0 ) ? argmax_val_0_0 : n_1_2;
    assign {argmax_idx_1_0} = ( cmp_1_0 ) ? argmax_idx_0_0 : 2'b10;

    assign out = argmax_idx_1_0;

endmodule

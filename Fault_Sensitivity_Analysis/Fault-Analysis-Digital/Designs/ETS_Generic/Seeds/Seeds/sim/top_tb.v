`timescale 1ns/1ps
`define EOF 32'hFFFF_FFFF
`define NULL 0

module top_tb();

    parameter OUTWIDTH=2;
    parameter NUM_A=7;
    parameter WIDTH_A=4;
    parameter NUM_W=30;
    parameter WIDTH_W=8;
    parameter NUM_B0=3;
    parameter WIDTH_B0=11;
    parameter NUM_B1=3;
    parameter WIDTH_B1=14;

localparam period = 320000000.00;

    reg  [WIDTH_A-1:0] at[NUM_A-1:0];
    wire [NUM_A*WIDTH_A-1:0] inp;
    wire [OUTWIDTH-1:0] out;

    wire [WIDTH_A:0] r;
wire [NUM_W*WIDTH_W-1:0] weights;
//weights = [-43, -76, 0, 23, -105, 2, -19, 110, 0, 49, -36, 48, 46, -23, 64, 58, -95, -13, -36, 109, 21, 6, 12, -7, -5, -10, -5, -9, -7, -6]
assign weights = {8'sb11010101, 8'sb10110100, 8'sb00000000, 8'sb00010111, 8'sb10010111, 8'sb00000010, 8'sb11101101, 8'sb01101110, 8'sb00000000, 8'sb00110001, 8'sb11011100, 8'sb00110000, 8'sb00101110, 8'sb11101001, 8'sb01000000, 8'sb00111010, 8'sb10100001, 8'sb11110011, 8'sb11011100, 8'sb01101101, 8'sb00010101, 8'sb00000110, 8'sb00001100, 8'sb11111001, 8'sb11111011, 8'sb11110110, 8'sb11111011, 8'sb11110111, 8'sb11111001, 8'sb11111010};
wire [(NUM_B0*WIDTH_B0)+(NUM_B1*WIDTH_B1)-1:0] biases;
//biases = [3343, -3411, -4678, -908, 73, -176]
assign biases = {14'sb00110100001111, 14'sb11001010101101, 14'sb10110110111010, 11'sb10001110100, 11'sb00001001001, 11'sb11101010000};
top DUT(.inp(inp),
        .weights(weights),
        .biases(biases),
        .out(out)
        );


    integer inFile,outFile,i;
    initial
    begin
        $display($time, " << Starting the Simulation >>");
            inFile = $fopen("./sim/sim.Xtest","r");
        if (inFile == `NULL) begin
                $display($time, " file not found");
                $finish;
        end
        outFile = $fopen("./sim/output.txt");
        while (!$feof(inFile)) begin
            for (i=0;i<NUM_A;i=i+1) begin
                $fscanf(inFile,"%d ",at[i]);
            end
            $fscanf(inFile,"\n");
            #(period)
            $fwrite(outFile,"%d\n", out);
        end
        #(period)
        $display($time, " << Finishing the Simulation >>");
        $fclose(outFile);
        $fclose(inFile);
        $finish;
    end


    genvar gi;
    generate
    for (gi=0;gi<NUM_A;gi=gi+1) begin : genbit
        assign inp[(gi+1)*WIDTH_A-1:gi*WIDTH_A] = at[gi];
    end
    endgenerate


    endmodule

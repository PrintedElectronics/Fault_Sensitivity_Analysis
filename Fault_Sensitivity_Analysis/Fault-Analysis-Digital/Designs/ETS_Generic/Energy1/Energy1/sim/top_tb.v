`timescale 1ns/1ps
`define EOF 32'hFFFF_FFFF
`define NULL 0

module top_tb();

    parameter OUTWIDTH=2;
    parameter NUM_A=8;
    parameter WIDTH_A=4;
    parameter NUM_W=33;
    parameter WIDTH_W=8;
    parameter NUM_B0=3;
    parameter WIDTH_B0=11;
    parameter NUM_B1=3;
    parameter WIDTH_B1=14;

localparam period = 250000000.00;

    reg  [WIDTH_A-1:0] at[NUM_A-1:0];
    wire [NUM_A*WIDTH_A-1:0] inp;
    wire [OUTWIDTH-1:0] out;

    wire [WIDTH_A:0] r;
wire [NUM_W*WIDTH_W-1:0] weights;
//weights = [-7, 5, -3, 7, 0, -4, 95, -19, 4, 4, -18, 0, 104, -78, -85, 1, -77, -1, 6, -1, 75, 44, 31, -18, 37, -4, -4, -5, -5, -2, -6, -1, -6]
assign weights = {8'sb11111001, 8'sb00000101, 8'sb11111101, 8'sb00000111, 8'sb00000000, 8'sb11111100, 8'sb01011111, 8'sb11101101, 8'sb00000100, 8'sb00000100, 8'sb11101110, 8'sb00000000, 8'sb01101000, 8'sb10110010, 8'sb10101011, 8'sb00000001, 8'sb10110011, 8'sb11111111, 8'sb00000110, 8'sb11111111, 8'sb01001011, 8'sb00101100, 8'sb00011111, 8'sb11101110, 8'sb00100101, 8'sb11111100, 8'sb11111100, 8'sb11111011, 8'sb11111011, 8'sb11111110, 8'sb11111010, 8'sb11111111, 8'sb11111010};
wire [(NUM_B0*WIDTH_B0)+(NUM_B1*WIDTH_B1)-1:0] biases;
//biases = [-5864, -579, 5138, 145, -576, -148]
assign biases = {14'sb10100100011000, 14'sb11110110111101, 14'sb01010000010010, 11'sb00010010001, 11'sb10111000000, 11'sb11101101100};
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

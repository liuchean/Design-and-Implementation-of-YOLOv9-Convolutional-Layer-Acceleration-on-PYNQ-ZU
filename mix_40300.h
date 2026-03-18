#include "ap_fixed.h"

////原本是18 6 可以改為16 6
//typedef ap_fixed<19, 7> fixed_t;//input
////主要是要改善input
//typedef ap_fixed<20, 7> fixed_tt;//output
//
//typedef ap_fixed<20, 7> fixed_ttt;//weight

//原本是18 6 可以改為16 6
typedef ap_fixed<20, 6> fixed_t;//input
//主要是要改善input
typedef ap_fixed<20, 6> fixed_tt;//output

typedef ap_fixed<20, 6> fixed_ttt;//weight

//輸入
#define bb 1
#define c1 48
#define input_height 40
#define input_width 30

//輸入權重
#define c2 48 //輸出通道
//c1
#define kernel_size_a 1
#define kernel_size_b 1

//輸出
// b
// c2
#define output_height 40
#define output_width 30


void mix_40300(
		const float input[bb][c1][input_height][input_width],
		const float weight[c2][c1][kernel_size_a][kernel_size_b],

		float output[bb][c2][output_height][output_width],
		float output1[bb][c2][output_height][output_width],
		float output2[bb][c2][output_height][output_width],
		float output3[bb][c2][output_height][output_width],
		float output4[bb][c2][output_height][output_width],
		int check1
);

void conv2d_2448(
		fixed_t local_input_part1[bb][c1/2][input_height][input_width],
		fixed_t local_input_part2[bb][c1/2][input_height][input_width],
		fixed_ttt local_weight_part1[c2][c1/2][kernel_size_a][kernel_size_b],
		fixed_ttt local_weight_part2[c2][c1/2][kernel_size_a][kernel_size_b],
		float local_output1[bb][c2][output_height][output_width],
		float local_output2[bb][c2][output_height][output_width],
		float local_output3[bb][c2][output_height][output_width],
		float local_output4[bb][c2][output_height][output_width]

);
void conv2d_4848(
		fixed_t local_input_part1[bb][c1/2][input_height][input_width],
		fixed_t local_input_part2[bb][c1/2][input_height][input_width],
		fixed_ttt local_weight_part1[c2][c1/2][kernel_size_a][kernel_size_b],
		fixed_ttt local_weight_part2[c2][c1/2][kernel_size_a][kernel_size_b],
		float local_output1[bb][c2][output_height][output_width],
		float local_output2[bb][c2][output_height][output_width],
		float local_output3[bb][c2][output_height][output_width],
		float local_output4[bb][c2][output_height][output_width]

);
void conv2d_2424(
		fixed_t local_input_part11[bb][24][input_height][input_width],
		fixed_t local_input_part22[bb][24][input_height][input_width],
		fixed_ttt local_weight_part1[48][24][kernel_size_a][kernel_size_b],
		fixed_ttt local_weight_part2[48][24][kernel_size_a][kernel_size_b],
		float local_output1[bb][c2][output_height][output_width],
		float local_output2[bb][c2][output_height][output_width],
		float local_output3[bb][c2][output_height][output_width],
		float local_output4[bb][c2][output_height][output_width]
);




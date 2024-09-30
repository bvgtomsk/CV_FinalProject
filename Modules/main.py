import pose_video_similarity_utils


def main(reference_video, video_to_compare, output, output_width = 1000, ref_start_ms=0, ref_end_ms=1e+10, comp_start_ms=0, comp_end_ms=1e+10, keypoints_threshold = 2, head_coef = 5, body_coef = 10, body_height = 500):
    ref_model = pose_video_similarity_utils.pose_video_model(reference_video, ref_start_ms, ref_end_ms, body_coef, head_coef, body_height, keypoints_threshold)
    comp_model = pose_video_similarity_utils.pose_video_model(video_to_compare, comp_start_ms, comp_end_ms, body_coef, head_coef, body_height, keypoints_threshold)
    comp_model.compare(ref_model)
    comp_model.generate_comparing_video(output, output_width)
    return 0

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--referance_video', metavar='path', required=True,
                        help='the path to reference video')
    parser.add_argument('--video_to_compare', metavar='path', required=True,
                        help='path to video wich you want to compare with reference')
    parser.add_argument('--output', metavar='path', required=True,
                        help='path to resulting video')
    parser.add_argument('--output_width', type = int, required=True,
                        help='width of output video')
    parser.add_argument('--ref_start_ms', type = int, required=False,
                        help='reference video time of start comparing')
    parser.add_argument('--ref_end_ms', type = int, required=False,
                        help='reference video time of end comparing')
    parser.add_argument('--comp_start_ms', type = int, required=False,
                        help='video to compare time of start comparing')
    parser.add_argument('--comp_end_ms', type = int, required=False,
                        help='video to compare time of end comparing')
    parser.add_argument('--keypoints_treshold', type = int, required=False,
                        help='keypoints threshold')
    parser.add_argument('--head_coef', type = int, required=False,
                        help='head coeficient')
    parser.add_argument('--body_coef', type = int, required=False,
                        help='body coeficient')
    parser.add_argument('--body_height', type = int, required=False,
                        help='body height - for transform keypoints orientire')
    args = parser.parse_args()
    main(reference_video=args.reference_video, 
         video_to_compare=args.video_to_compare, 
         output=args.output,
         output_width=args.output_width,
         ref_start_ms=args.ref_start_ms, 
         ref_end_ms=args.ref_end_ms, 
         comp_start_ms=args.comp_start_ms, 
         comp_end_ms=args.comp_end_ms, 
         keypoints_threshold=args.keypoints_threshold, 
         head_coef=args.head_coef, 
         body_coef=args.body_coef, 
         body_height=args.body_height)
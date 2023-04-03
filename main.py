import argparse
from algoritmes import TRAOne

def main(args):
    filter_classes = args.filter_classes

    if filter_classes:
        filter_classes = [filter_classes]

    dt_obj = TRAOne(
        tracker=0,
        detector=36,
        weights=args.weights,
        use_cuda=args.use_cuda
        )
    # Получить функцию отслеживания
    track_fn = dt_obj.track_video(args.video_path,
                                output_dir=args.output_dir,
                                conf_thres=args.conf_thres,
                                iou_thres=args.iou_thres,
                                display=args.display,
                                draw_trails=args.draw_trails,
                                filter_classes=filter_classes,
                                class_names=None) # class_names=['Номерной знак'] для пользовательских весов

    # Цикл по track_fn для извлечения выходных данных каждого кадра
    for bbox_details, frame_details in track_fn:
        bbox_xyxy, ids, scores, class_ids = bbox_details
        frame, frame_num, fps = frame_details
        print(frame_num)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('video_path', help='Путь к входному видео')
    parser.add_argument('--cpu', default=True, action='store_false', dest='use_cuda',
                        help='запуск на процессоре, если не указано иное, программа будет работать на графическом процессоре.')
    parser.add_argument('--no_save', default=True, action='store_false',
                        dest='save_result', help='сохранять результаты или нет')
    parser.add_argument('--no_display', default=True, action='store_false',
                        dest='display', help='отображать результаты на экране или нет')
    parser.add_argument('--output_dir', default='data/results',  help='Path to output directory')
    parser.add_argument('--draw_trails', action='store_true', default=False,
                        help='если предусмотрено, будут нарисованы трассы движения объекта.')
    parser.add_argument('--filter_classes', default=None, help='Filter class name')
    parser.add_argument('-w', '--weights', default=None, help='Траектория движения тренированных весов')
    parser.add_argument('-ct', '--conf_thres', default=0.25, type=float, help='пороговый показатель достоверности')
    parser.add_argument('-it', '--iou_thres', default=0.45, type=float, help='iou score threshold')

    args = parser.parse_args()

    main(args)
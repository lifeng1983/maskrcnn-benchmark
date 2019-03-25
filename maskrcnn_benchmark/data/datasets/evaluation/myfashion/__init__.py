import logging

from .myfashion_eval import do_myfashion_evaluation


def myfashion_evaluation(dataset, predictions, output_folder, box_only, **_):
    logger = logging.getLogger("maskrcnn_benchmark.inference")
    #if box_only:
    #    logger.warning("myfashion evaluation doesn't support box_only, ignored.")
    #logger.info("performing voc evaluation, ignored iou_types.")
    return do_myfashion_evaluation(
        dataset=dataset,
        predictions=predictions,
        output_folder=output_folder,
        logger=logger,
    )

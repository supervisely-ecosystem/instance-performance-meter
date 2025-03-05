from __future__ import annotations

import os
from copy import deepcopy
from datetime import datetime
from time import monotonic
from typing import Any, Callable, Dict, List, Literal, Optional, Union

import nbformat as nbf
import numpy as np
from supervisely import logger
from supervisely._utils import batched
from supervisely.api.annotation_api import (
    BITMAP,
    AlphaMask,
    Annotation,
    AnnotationApi,
    AnnotationJsonFields,
    LabelJsonFields,
    deepcopy,
)
from supervisely.api.api import Api
from supervisely.api.image_api import (
    SUPPORTED_CONFLICT_RESOLUTIONS,
    ApiField,
    HTTPError,
    ImageApi,
    ImageInfo,
    ProjectMeta,
    compare_dicts,
    get_file_ext,
    get_file_name,
)
from supervisely.api.project_api import ProjectApi, ProjectInfo
from supervisely.io.fs import mkdir
from supervisely.project.project import (
    ApiContext,
    Dataset,
    OpenMode,
    Project,
    ProjectType,
    TagCollection,
    clean_dir,
    create_readme,
    dump_json_file,
)
from supervisely.sly_logger import logger
from tabulate import tabulate
from tqdm import tqdm


class PerformanceStats:

    def __init__(self, name: str, num_items: int, process: str = None, project=False):
        self.project = project
        self.name = name
        self.process = process
        self._num_items = num_items
        self.start_time = 0
        self.last_time = 0
        self.total_time = 0
        self.batch_times = []
        self.full_times = []
        self._full_name = f"{self.process} {self.name}" if self.process else self.name
        self.summary = []
        self.overall_summary = {}

    @property
    def num_items(self):
        return self._num_items

    @num_items.setter
    def num_items(self, num_items: int):
        self._num_items = num_items

    def change_name(self, name: str = None, process: str = None):
        if name is None and process is None:
            raise ValueError("At least one of the arguments should be provided.")
        if name:
            self.name = name
        if process:
            self.process = process
        self._full_name = f"{self.process} {self.name}" if self.process else self.name

    def start(self):
        self.batch_times = []
        self.start_time = monotonic()
        self.last_time = self.start_time

    def update(self):
        current_time = monotonic()
        elapsed_time = current_time - self.last_time
        self.batch_times.append(elapsed_time)
        self.last_time = current_time

    def finalize(self):
        self.total_time = monotonic() - self.start_time
        self.full_times.append(self.total_time)

    def avg_items_per_second(self):
        return self._num_items / self.total_time if self.total_time > 0 else 0

    def percentile(self, percentile: float):
        return np.percentile(self.batch_times, percentile) if self.batch_times else 0

    def reset(self):
        self.start_time = 0
        self.last_time = 0
        self.total_time = 0
        self._num_items = 0
        self.batch_times = []

    def reset_full_times(self):
        self.full_times = []

    def reset_summary(self):
        self.summary = []

    def generate_summary(self):

        if self.project:
            summary = {
                "name": self._full_name,
                "iteration": self._num_items,
                "time": round(self.total_time, 2),
                "min_time": round(min(self.full_times, default=0), 2),
                "max_time": round(max(self.full_times, default=0), 2),
            }
        else:
            summary = {
                "name": self._full_name,
                "num_items": self._num_items,
                "total_time": round(self.total_time, 2),
                "avg_items_per_second": round(self.avg_items_per_second(), 2),
                "min_batch_time": round(min(self.batch_times, default=0), 2),
                "max_batch_time": round(max(self.batch_times, default=0), 2),
                "1th_percentile": round(self.percentile(1), 2),
                "10th_percentile": round(self.percentile(10), 2),
                "60th_percentile": round(self.percentile(60), 2),
                "70th_percentile": round(self.percentile(70), 2),
                "80th_percentile": round(self.percentile(80), 2),
                "90th_percentile": round(self.percentile(90), 2),
            }

        return summary

    def generate_performance_summary(self):
        if self.project:
            min_run_time = round(min(self.full_times, default=0), 2)
            max_run_time = round(max(self.full_times, default=0), 2)
            avg_run_time = round(np.mean(self.full_times or 0), 2)
            stddev_run_time = round(np.std(self.full_times or 0), 2)
            overall_time = round(sum(self.full_times), 2)
            summary = {
                "name": self._full_name,
                "total_iterations": self._num_items,
                "total_run_time": overall_time,
                "min_run_time": min_run_time,
                "max_run_time": max_run_time,
                "avg_run_time": avg_run_time,
                "stddev_run_time": stddev_run_time,
            }
        else:
            min_run_time = round(min(self.full_times, default=0), 2)
            max_run_time = round(max(self.full_times, default=0), 2)
            avg_run_time = round(np.mean(self.full_times or 0), 2)
            stddev_run_time = round(np.std(self.full_times or 0), 2)
            summary = {
                "name": self._full_name,
                "num_items": self._num_items,
                "min_run_time": min_run_time,
                "max_run_time": max_run_time,
                "avg_run_time": avg_run_time,
                "stddev_run_time": stddev_run_time,
            }
        return summary

    def log_summary(self):
        summary = self.generate_summary()
        self.summary.append(deepcopy(summary))
        name = summary.pop("name")
        if all(value == 0 for value in summary.values() if isinstance(value, (int, float))):
            logger.info(f"{name} Iteration Performance: No data to display.")
            return
        table = tabulate(summary.items(), headers=["Metric", "Value"], tablefmt="grid")
        logger.info(f"{name} Iteration Performance:")
        logger.info(f"{table}\n")

    def log_overall_summary(self):
        summary = self.generate_performance_summary()
        self.overall_summary = deepcopy(summary)
        name = summary.pop("name")
        if all(value == 0 for value in summary.values() if isinstance(value, (int, float))):
            logger.info(f"{name} Performance Summary: No data to display.")
            return
        table = tabulate(summary.items(), headers=["Metric", "Value"], tablefmt="grid")
        logger.info(f"{name} Performance Summary:")
        logger.info(f"{table}\n")


class ImageApiWithStats(ImageApi):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stats_binary = PerformanceStats("Image Binaries", 0)
        self.stats_hashes = PerformanceStats("Image Hashes", 0)

    def _upload_data_bulk(
        self, func_item_to_byte_stream, items_hashes, retry_cnt=3, progress_cb=None
    ):
        # count all items to adjust progress_cb and create hash to item mapping with unique hashes
        items_count_total = 0
        hash_to_items = {}
        for item, i_hash in items_hashes:
            hash_to_items[i_hash] = item
            items_count_total += 1

        unique_hashes = set(hash_to_items.keys())
        remote_hashes = set(
            self.check_existing_hashes(list(unique_hashes))
        )  # existing -- from server
        if progress_cb:
            progress_cb(len(remote_hashes))
        pending_hashes = unique_hashes - remote_hashes

        for retry_idx in range(retry_cnt):
            # single attempt to upload all data which is not uploaded yet
            self.stats_binary.start()
            for hashes in batched(list(pending_hashes)):
                pending_hashes_items = [(h, hash_to_items[h]) for h in hashes]
                self.stats_binary.num_items = self.stats_binary.num_items + len(
                    pending_hashes_items
                )
                hashes_rcv = self._upload_uniq_images_single_req(
                    func_item_to_byte_stream, pending_hashes_items
                )
                self.stats_binary.update()
                pending_hashes -= set(hashes_rcv)
                if set(hashes_rcv) - set(hashes):
                    logger.warning(
                        "Hash inconsistency in images bulk upload.",
                        extra={"sent": hashes, "received": hashes_rcv},
                    )
                if progress_cb:
                    progress_cb(len(hashes_rcv))

            self.stats_binary.finalize()
            self.stats_binary.log_summary()
            if not pending_hashes:
                if progress_cb is not None:
                    progress_cb(items_count_total - len(unique_hashes))
                return

            warning_items = []
            for h in pending_hashes:
                item_data = hash_to_items[h]
                if isinstance(item_data, (bytes, bytearray)):
                    item_data = "some bytes ..."
                warning_items.append((h, item_data))

            logger.warning(
                "Unable to upload images (data).",
                extra={
                    "retry_idx": retry_idx,
                    "items": warning_items,
                },
            )
            # now retry it for the case if it is a shadow server/connection error

        raise ValueError(
            "Unable to upload images (data). "
            "Please check if images are in supported format and if ones aren't corrupted."
        )

    def _upload_bulk_add(
        self,
        func_item_to_kv,
        dataset_id,
        names,
        items,
        progress_cb=None,
        metas=None,
        batch_size=50,
        force_metadata_for_links=True,
        skip_validation=False,
        conflict_resolution: Optional[Literal["rename", "skip", "replace"]] = None,
        validate_meta: Optional[bool] = False,
        use_strict_validation: Optional[bool] = False,
        use_caching_for_validation: Optional[bool] = False,
    ):
        """ """
        if use_strict_validation and not validate_meta:
            raise ValueError(
                "use_strict_validation is set to True, while validate_meta is set to False. "
                "Please set validate_meta to True to use strict validation "
                "or disable strict validation by setting use_strict_validation to False."
            )
        if validate_meta:
            dataset_info = self._api.dataset.get_info_by_id(dataset_id)

            validation_schema = self._api.project.get_validation_schema(
                dataset_info.project_id, use_caching=use_caching_for_validation
            )

            if validation_schema is None:
                raise ValueError(
                    "Validation schema is not set for the project, while "
                    "validate_meta is set to True. Either disable the validation "
                    "or set the validation schema for the project using the "
                    "api.project.set_validation_schema method."
                )

            for idx, meta in enumerate(metas):
                missing_fields, extra_fields = compare_dicts(
                    validation_schema, meta, strict=use_strict_validation
                )

                if missing_fields or extra_fields:
                    raise ValueError(
                        f"Validation failed for the metadata of the image with index {idx} and name {names[idx]}. "
                        "Please check the metadata and try again. "
                        f"Missing fields: {missing_fields}, Extra fields: {extra_fields}"
                    )

        if (
            conflict_resolution is not None
            and conflict_resolution not in SUPPORTED_CONFLICT_RESOLUTIONS
        ):
            raise ValueError(
                f"Conflict resolution should be one of the following: {SUPPORTED_CONFLICT_RESOLUTIONS}"
            )
        if len(set(names)) != len(names):
            raise ValueError("Some image names are duplicated, only unique images can be uploaded.")

        results = []

        def _add_timestamp(name: str) -> str:
            now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            return f"{get_file_name(name)}_{now}{get_file_ext(name)}"

        def _pack_for_request(names: List[str], items: List[Any], metas: List[Dict]) -> List[Any]:
            images = []
            for name, item, meta in zip(names, items, metas):
                item_tuple = func_item_to_kv(item)
                image_data = {ApiField.TITLE: name, item_tuple[0]: item_tuple[1]}
                if hasattr(self, "sort_by") and self.sort_by is not None:
                    meta = self._add_custom_sort(meta, name)
                if len(meta) != 0 and type(meta) == dict:
                    image_data[ApiField.META] = meta
                images.append(image_data)
            return images

        if len(names) == 0:
            return results
        if len(names) != len(items):
            raise ValueError('Can not match "names" and "items" lists, len(names) != len(items)')

        if metas is None:
            metas = [{}] * len(names)
        else:
            if len(names) != len(metas):
                raise ValueError('Can not match "names" and "metas" len(names) != len(metas)')

        idx_to_id = {}
        self.stats_hashes.start()
        for batch_count, (batch_names, batch_items, batch_metas) in enumerate(
            zip(
                batched(names, batch_size=batch_size),
                batched(items, batch_size=batch_size),
                batched(metas, batch_size=batch_size),
            )
        ):
            self.stats_hashes.num_items = self.stats_hashes.num_items + len(batch_names)
            for retry in range(2):
                images = _pack_for_request(batch_names, batch_items, batch_metas)
                try:
                    response = self._api.post(
                        "images.bulk.add",
                        {
                            ApiField.DATASET_ID: dataset_id,
                            ApiField.IMAGES: images,
                            ApiField.FORCE_METADATA_FOR_LINKS: force_metadata_for_links,
                            ApiField.SKIP_VALIDATION: skip_validation,
                        },
                    )
                    if progress_cb is not None:
                        progress_cb(len(images))

                    for info_json in response.json():
                        info_json_copy = info_json.copy()
                        if info_json.get(ApiField.MIME, None) is not None:
                            info_json_copy[ApiField.EXT] = info_json[ApiField.MIME].split("/")[1]
                        results.append(self._convert_json_info(info_json_copy))
                    break
                except HTTPError as e:
                    error_details = e.response.json().get("details", {})
                    if isinstance(error_details, list):
                        error_details = error_details[0]
                    if (
                        conflict_resolution is not None
                        and e.response.status_code == 400
                        and error_details.get("type") == "NONUNIQUE"
                    ):
                        logger.info(
                            f"Handling the exception above with '{conflict_resolution}' conflict resolution method"
                        )

                        errors: List[Dict] = error_details.get("errors", [])

                        if conflict_resolution == "replace":
                            ids_to_remove = [error["id"] for error in errors]
                            logger.info(f"Image ids to be removed: {ids_to_remove}")
                            self._api.image.remove_batch(ids_to_remove)
                            continue

                        name_to_index = {name: idx for idx, name in enumerate(batch_names)}
                        errors = sorted(
                            errors, key=lambda x: name_to_index[x["name"]], reverse=True
                        )
                        if conflict_resolution == "rename":
                            for error in errors:
                                error_img_name = error["name"]
                                idx = name_to_index[error_img_name]
                                batch_names[idx] = _add_timestamp(error_img_name)
                        elif conflict_resolution == "skip":
                            for error in errors:
                                error_img_name = error["name"]
                                error_index = name_to_index[error_img_name]

                                idx_to_id[error_index + batch_count * batch_size] = error["id"]
                                for l in [batch_items, batch_metas, batch_names]:
                                    l.pop(error_index)

                        if len(batch_names) == 0:
                            break
                    else:
                        raise
            self.stats_hashes.update()
        if len(idx_to_id) > 0:
            logger.info(
                "Adding ImageInfo of images with the same name that already exist in the dataset to the response."
            )
            idx_to_id = dict(reversed(list(idx_to_id.items())))
            image_infos = self._api.image.get_info_by_id_batch(list(idx_to_id.values()))
            for idx, info in zip(list(idx_to_id.values()), image_infos):
                results.insert(idx, info)
        self.stats_hashes.finalize()
        self.stats_hashes.log_summary()
        return results  # ordered_results


class AnnotationApiWithStats(AnnotationApi):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stats = PerformanceStats("Image Annotations", 0)

    def _upload_batch(
        self,
        func_ann_to_json: Callable,
        img_ids: List[int],
        anns: List[Union[Dict, Annotation, str]],
        progress_cb=None,
        skip_bounds_validation: Optional[bool] = False,
    ):

        # img_ids from the same dataset
        if len(img_ids) == 0:
            return
        if len(img_ids) != len(anns):
            raise RuntimeError(
                'Can not match "img_ids" and "anns" lists, len(img_ids) != len(anns)'
            )

        # use context to avoid redundant API calls
        dataset_id = self._api.image.get_info_by_id(
            img_ids[0], force_metadata_for_links=False
        ).dataset_id
        context = self._api.optimization_context
        context_dataset_id = context.get("dataset_id")
        project_id = context.get("project_id")
        project_meta = context.get("project_meta")
        if dataset_id != context_dataset_id:
            context["dataset_id"] = dataset_id
            project_id, project_meta = None, None

        if not isinstance(project_meta, ProjectMeta):
            if project_id is None:
                project_id = self._api.dataset.get_info_by_id(dataset_id).project_id
                context["project_id"] = project_id
            project_meta = ProjectMeta.from_json(self._api.project.get_meta(project_id))
            context["project_meta"] = project_meta

        need_upload_alpha_masks = False
        for obj_cls in project_meta.obj_classes:
            if obj_cls.geometry_type == AlphaMask:
                need_upload_alpha_masks = True
                break
        self.stats.start()
        for batch in batched(list(zip(img_ids, anns))):
            self.stats.num_items = self.stats.num_items + len(batch)
            data = []
            if need_upload_alpha_masks:
                special_figures = []
                special_geometries = []
                # check if there are any AlphaMask geometries in the batch
                for img_id, ann in batch:
                    ann_json = func_ann_to_json(ann)
                    ann_json = deepcopy(ann_json)  # Avoid changing the original data

                    ann_json = Annotation._to_subpixel_coordinate_system_json(ann_json)
                    filtered_labels = []
                    if AnnotationJsonFields.LABELS not in ann_json:
                        raise RuntimeError(
                            f"Annotation JSON does not contain '{AnnotationJsonFields.LABELS}' field"
                        )
                    for label_json in ann_json[AnnotationJsonFields.LABELS]:
                        for key in [
                            LabelJsonFields.GEOMETRY_TYPE,
                            LabelJsonFields.OBJ_CLASS_NAME,
                        ]:
                            if key not in label_json:
                                raise RuntimeError(f"Label JSON does not contain '{key}' field")
                        if label_json[LabelJsonFields.GEOMETRY_TYPE] == AlphaMask.geometry_name():
                            label_json.update({ApiField.ENTITY_ID: img_id})

                            obj_cls_name = label_json.get(LabelJsonFields.OBJ_CLASS_NAME)
                            obj_cls = project_meta.get_obj_class(obj_cls_name)
                            if obj_cls is None:
                                raise RuntimeError(
                                    f"Object class '{obj_cls_name}' not found in project meta"
                                )
                            # update obj class id in label json
                            label_json[LabelJsonFields.OBJ_CLASS_ID] = obj_cls.sly_id

                            geometry = label_json.pop(
                                BITMAP
                            )  # remove alpha mask geometry from label json
                            special_geometries.append(geometry)
                            special_figures.append(label_json)
                        else:
                            filtered_labels.append(label_json)
                    if len(filtered_labels) != len(ann_json[AnnotationJsonFields.LABELS]):
                        ann_json[AnnotationJsonFields.LABELS] = filtered_labels
                    data.append({ApiField.IMAGE_ID: img_id, ApiField.ANNOTATION: ann_json})
            else:
                for img_id, ann in batch:
                    ann_json = func_ann_to_json(ann)
                    ann_json = deepcopy(ann_json)  # Avoid changing the original data
                    ann_json = Annotation._to_subpixel_coordinate_system_json(ann_json)
                    data.append({ApiField.IMAGE_ID: img_id, ApiField.ANNOTATION: ann_json})

            self._api.post(
                "annotations.bulk.add",
                data={
                    ApiField.DATASET_ID: dataset_id,
                    ApiField.ANNOTATIONS: data,
                    ApiField.SKIP_BOUNDS_VALIDATION: skip_bounds_validation,
                },
            )
            if need_upload_alpha_masks:
                if len(special_figures) > 0:
                    # 1. create figures
                    json_body = {
                        ApiField.DATASET_ID: dataset_id,
                        ApiField.FIGURES: special_figures,
                        ApiField.SKIP_BOUNDS_VALIDATION: skip_bounds_validation,
                    }
                    resp = self._api.post("figures.bulk.add", json_body)
                    added_fig_ids = [resp_obj[ApiField.ID] for resp_obj in resp.json()]

                    # 2. upload alpha mask geometries
                    self._api.image.figure.upload_geometries_batch(
                        added_fig_ids, special_geometries
                    )

            if progress_cb is not None:
                progress_cb(len(batch))
            self.stats.update()
        self.stats.finalize()
        self.stats.log_summary()


class ProjectApiHeaderless(ProjectApi):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def create(
        self: ProjectApi,
        workspace_id: int,
        name: str,
        type: ProjectType = ProjectType.IMAGES,
        description: Optional[str] = "",
        change_name_if_conflict: Optional[bool] = False,
    ) -> ProjectInfo:
        """
        The same as ProjectApi.create, but without x-task-id header

        """
        effective_name = self._get_effective_new_name(
            parent_id=workspace_id,
            name=name,
            change_name_if_conflict=change_name_if_conflict,
        )
        task_id_header = self._api.headers.pop("x-task-id", None)
        response = self._api.post(
            "projects.add",
            {
                ApiField.WORKSPACE_ID: workspace_id,
                ApiField.NAME: effective_name,
                ApiField.DESCRIPTION: description,
                ApiField.TYPE: str(type),
            },
        )
        self._api.headers["x-task-id"] = task_id_header
        return self._convert_json_info(response.json())


class ApiWithStats(Api):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image = ImageApiWithStats(self)
        self.annotation = AnnotationApiWithStats(self)
        self.project = ProjectApiHeaderless(self)
        self.stats_project_upload = PerformanceStats("Upload Project", 0, project=True)
        self.stats_project_download = PerformanceStats("Download Project", 0, project=True)


def _download_project(
    api: ApiWithStats,
    project_id: int,
    dest_dir: str,
    dataset_ids: Optional[List[int]] = None,
    log_progress: bool = True,
    batch_size: Optional[int] = 50,
    only_image_tags: Optional[bool] = False,
    save_image_info: Optional[bool] = False,
    save_images: Optional[bool] = True,
    progress_cb: Optional[Callable] = None,
    save_image_meta: Optional[bool] = False,
    images_ids: Optional[List[int]] = None,
    resume_download: Optional[bool] = False,
):
    dataset_ids = set(dataset_ids) if (dataset_ids is not None) else None
    project_fs = None
    meta = ProjectMeta.from_json(api.project.get_meta(project_id, with_settings=True))
    if os.path.exists(dest_dir) and resume_download:
        dump_json_file(meta.to_json(), os.path.join(dest_dir, "meta.json"))
        try:
            project_fs = Project(dest_dir, OpenMode.READ)
        except RuntimeError as e:
            if "Project is empty" in str(e):
                clean_dir(dest_dir)
                project_fs = None
            else:
                raise
    if project_fs is None:
        project_fs = Project(dest_dir, OpenMode.CREATE)
    project_fs.set_meta(meta)

    if progress_cb is not None:
        log_progress = False

    id_to_tagmeta = None
    if only_image_tags is True:
        id_to_tagmeta = meta.tag_metas.get_id_mapping()

    existing_datasets = {dataset.path: dataset for dataset in project_fs.datasets}
    for parents, dataset in api.dataset.tree(project_id):
        dataset_path = Dataset._get_dataset_path(dataset.name, parents)
        dataset_id = dataset.id
        if dataset_ids is not None and dataset_id not in dataset_ids:
            continue

        if dataset_path in existing_datasets:
            dataset_fs = existing_datasets[dataset_path]
        else:
            dataset_fs = project_fs.create_dataset(dataset.name, dataset_path)

        all_images = api.image.get_list(dataset_id, force_metadata_for_links=False)
        images = [image for image in all_images if images_ids is None or image.id in images_ids]
        ds_total = len(images)

        ds_progress = progress_cb
        if log_progress is True:
            ds_progress = tqdm(
                desc="Downloading images from {!r}".format(dataset.name),
                total=ds_total,
            )

        anns_progress = None
        if log_progress or progress_cb is not None:
            anns_progress = tqdm(
                desc="Downloading annotations from {!r}".format(dataset.name),
                total=ds_total,
                leave=False,
            )

        with ApiContext(
            api,
            project_id=project_id,
            dataset_id=dataset_id,
            project_meta=meta,
        ):
            iterations = round(len(images) / batch_size)
            logger.info(f"Downloading {ds_total} dataset items in {iterations} iterations")
            iteration = 1
            for batch in batched(images, batch_size):
                logger.debug(f"Iteration {iteration}/{iterations}")
                image_ids = [image_info.id for image_info in batch]
                image_names = [image_info.name for image_info in batch]

                existing_image_infos: Dict[str, ImageInfo] = {}
                for image_name in image_names:
                    try:
                        image_info = dataset_fs.get_item_info(image_name)
                    except:
                        image_info = None
                    existing_image_infos[image_name] = image_info

                indexes_to_download = []
                for i, image_info in enumerate(batch):
                    existing_image_info = existing_image_infos[image_info.name]
                    if (
                        existing_image_info is None
                        or existing_image_info.updated_at != image_info.updated_at
                    ):
                        indexes_to_download.append(i)

                # download images in numpy format
                batch_imgs_bytes = [None] * len(image_ids)
                if save_images and indexes_to_download:
                    api.image.stats_binary.num_items += len(indexes_to_download)
                    if iteration == 1:
                        api.image.stats_binary.start()
                    for index, img in zip(
                        indexes_to_download,
                        api.image.download_bytes(
                            dataset_id,
                            [image_ids[i] for i in indexes_to_download],
                            progress_cb=ds_progress,
                        ),
                    ):
                        batch_imgs_bytes[index] = img
                    if iteration < iterations:
                        api.image.stats_binary.update()
                    else:
                        api.image.stats_binary.finalize()
                        api.image.stats_binary.log_summary()
                if ds_progress is not None:
                    ds_progress(len(batch) - len(indexes_to_download))

                # download annotations in json format
                ann_jsons = [None] * len(image_ids)
                if only_image_tags is False:
                    api.annotation.stats.num_items += len(indexes_to_download)
                    if iteration == 1:
                        api.annotation.stats.start()
                    if indexes_to_download:
                        for index, ann_info in zip(
                            indexes_to_download,
                            api.annotation.download_batch(
                                dataset_id,
                                [image_ids[i] for i in indexes_to_download],
                                progress_cb=anns_progress,
                            ),
                        ):
                            ann_jsons[index] = ann_info.annotation
                    if iteration < iterations:
                        api.annotation.stats.update()
                    else:
                        api.annotation.stats.finalize()
                        api.annotation.stats.log_summary()
                else:
                    if indexes_to_download:
                        for index in indexes_to_download:
                            image_info = batch[index]
                            tags = TagCollection.from_api_response(
                                image_info.tags,
                                meta.tag_metas,
                                id_to_tagmeta,
                            )
                            tmp_ann = Annotation(
                                img_size=(image_info.height, image_info.width), img_tags=tags
                            )
                            ann_jsons[index] = tmp_ann.to_json()
                            if anns_progress is not None:
                                anns_progress(len(indexes_to_download))
                if anns_progress is not None:
                    anns_progress(len(batch) - len(indexes_to_download))

                for img_info, name, img_bytes, ann in zip(
                    batch, image_names, batch_imgs_bytes, ann_jsons
                ):
                    dataset_fs: Dataset
                    # to fix already downloaded images that doesn't have info files
                    dataset_fs.delete_item(name)
                    dataset_fs.add_item_raw_bytes(
                        item_name=name,
                        item_raw_bytes=img_bytes if save_images is True else None,
                        ann=dataset_fs.get_ann(name, meta) if ann is None else ann,
                        img_info=img_info if save_image_info is True else None,
                    )
                iteration += 1
        if save_image_meta:
            meta_dir = dataset_fs.meta_dir
            for image_info in images:
                if image_info.meta:
                    mkdir(meta_dir)
                    dump_json_file(image_info.meta, dataset_fs.get_item_meta_path(image_info.name))

        # delete redundant items
        items_names_set = set([img.name for img in all_images])
        for item_name in dataset_fs.get_items_names():
            if item_name not in items_names_set:
                dataset_fs.delete_item(item_name)
    try:
        create_readme(dest_dir, project_id, api)
    except Exception as e:
        logger.info(f"There was an error while creating README: {e}")


def generate_tables_from_iterations(info: dict[List[List[dict]]]) -> str:
    html_content = ""
    for process, iterations in info.items():

        html_content += f"<h1>{process.title()}</h1>\n"

        for iteration_index, iteration in enumerate(iterations, start=1):
            if iteration_index == len(iterations):
                html_content += "<h2>Summary</h2>\n"
            else:
                html_content += f"<h2>Iteration {iteration_index}</h2>\n"
            html_content += '<div style="display: flex; justify-content: space-between; padding-bottom: 20px;">\n'

            for summary in iteration:
                if summary.get("name", None) is None:
                    logger.info(f"Skipping summary without name: {iteration}")
                    continue
                table_title = summary.pop("name")
                # create table header
                html_content += '<table style="width: 30%; margin-right: 20px;">\n'
                html_content += f'<tr><td colspan="2" style="text-align: center; font-weight: bold;">{table_title}</td></tr>\n'
                html_content += "<tr><th>Metric</th><th>Value</th></tr>\n"

                # add table data
                for key, value in summary.items():
                    #  format column names
                    column_name = key.replace("_", " ").title()
                    html_content += f"<tr><td>{column_name}</td><td>{value}</td></tr>\n"

                # close table
                html_content += "</table>\n"

            html_content += "</div>\n"

    return html_content


def create_ipynb_file(info: dict[List[List[dict]]], filename: str):
    # generate HTML
    html_output = generate_tables_from_iterations(info)

    # create new block for Jupyter Notebook
    nb = nbf.v4.new_notebook()

    # add markdown cell with HTML content
    nb.cells.append(nbf.v4.new_markdown_cell(html_output))

    # save Jupyter Notebook to file
    with open(filename, "w", encoding="utf-8") as f:
        nbf.write(nb, f)

import os

from dotenv import load_dotenv
from supervisely import is_development, logger
from supervisely.io.fs import clean_dir, dir_exists, remove_dir, unpack_archive
from supervisely.project.project import upload_project

from utils import ApiWithStats, _download_project, create_ipynb_file

if is_development():
    load_dotenv("~/supersivisely.env")
    load_dotenv("local.env")

workspace_id = int(os.getenv("WORKSPACE_ID"))
number_of_iterations = int(os.getenv("modal.state.iterations", 10))

data_dir_name = "/tmp/Extracted_Performance_Test_Project"

api: ApiWithStats = ApiWithStats.from_env()

workspace_info = api.workspace.get_info_by_id(workspace_id)
report_file = f"{api.task_id}_Performance_Test_Report.ipynb"

logger.info(f"Instance perfromance meter is ready")


def measure_upload():
    logger.info(f"Starting upload project test")
    unpack_archive("/tmp/Performance_Test.tar", data_dir_name)
    api.image.stats_binary.change_name(process="Upload")
    api.image.stats_hashes.change_name(process="Upload")
    api.annotation.stats.change_name(process="Upload")
    if not dir_exists(data_dir_name):
        logger.error(f"Data directory {data_dir_name} does not exist")
        return
    for i in range(1, number_of_iterations + 1):
        api.stats_project_upload.start()
        if i > 1:
            api.image.stats_binary.reset()
            api.image.stats_hashes.reset()
            api.annotation.stats.reset()
        api.stats_project_upload.num_items = i
        logger.info(f"Iteration {i} of {number_of_iterations}")
        project_name = f"Upload Performance Test Project {i}"
        proj_id, _ = upload_project(
            dir=data_dir_name,
            api=api,
            workspace_id=workspace_id,
            project_name=project_name,
            log_progress=True,
        )
        api.stats_project_upload.finalize()
        api.stats_project_upload.log_summary()
        if i != number_of_iterations:
            api.project.remove(proj_id)
    logger.info(f"Upload project test finished")
    api.image.stats_binary.log_overall_summary()
    api.image.stats_hashes.log_overall_summary()
    api.annotation.stats.log_overall_summary()
    api.stats_project_upload.log_overall_summary()
    iterations = []
    for binary, hashes, anns, project in zip(
        api.image.stats_binary.summary,
        api.image.stats_hashes.summary,
        api.annotation.stats.summary,
        api.stats_project_upload.summary,
    ):
        iterations.append([binary, hashes, anns, project])
    iterations.append(
        [
            api.image.stats_binary.overall_summary,
            api.image.stats_hashes.overall_summary,
            api.annotation.stats.overall_summary,
            api.stats_project_upload.overall_summary,
        ]
    )
    return iterations, proj_id


def measure_download(project_id: int):
    remove_dir(data_dir_name)
    logger.info(f"Starting download project test")
    api.image.stats_binary.change_name(process="Download")
    api.annotation.stats.change_name(process="Download")
    api.image.stats_binary.reset_full_times()
    api.annotation.stats.reset_full_times()
    api.image.stats_binary.reset_summary()
    api.annotation.stats.reset_summary()
    for i in range(1, number_of_iterations + 1):
        api.image.stats_binary.reset()
        api.annotation.stats.reset()
        api.stats_project_download.start()
        api.stats_project_download.num_items = i
        logger.info(f"Iteration {i} of {number_of_iterations}")
        _download_project(api, project_id, data_dir_name)
        api.stats_project_download.finalize()
        api.stats_project_download.log_summary()
        remove_dir(data_dir_name)
    logger.info(f"Download project test finished")
    api.project.remove(project_id)
    api.image.stats_binary.log_overall_summary()
    api.annotation.stats.log_overall_summary()
    api.stats_project_download.log_overall_summary()
    iterations = []
    for binary, anns, project in zip(
        api.image.stats_binary.summary,
        api.annotation.stats.summary,
        api.stats_project_download.summary,
    ):
        iterations.append([binary, anns, project])
    iterations.append(
        [
            api.image.stats_binary.overall_summary,
            api.annotation.stats.overall_summary,
            api.stats_project_download.overall_summary,
        ]
    )

    return iterations


if __name__ == "__main__":
    try:
        iterations_u, project_id = measure_upload()
        clean_dir(data_dir_name)
        iterations_d = measure_download(project_id)
        create_ipynb_file({"upload": iterations_u, "download": iterations_d}, report_file)
        file_info = api.file.upload(
            team_id=workspace_info.team_id,
            src=report_file,
            dst=report_file,
        )
        api.task.set_output_report(
            task_id=api.task_id,
            file_id=file_info.id,
            file_name=report_file,
            description="Report with performance test results. You could open it right in Team Files",
        )
    except Exception as e:
        api.task.set_output_text(
            task_id=api.task_id, title="Performance test failed", description="See logs for details"
        )

"""Download helpers for FSC-147 and SAM weights.

FSC-147 is hosted on Google Drive; automation requires `gdown`. This module
prints the exact commands and URLs so the user can execute them manually
(or via `gdown` if installed). Nothing is downloaded implicitly.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent


FSC147_INSTRUCTIONS = dedent("""\
    FSC-147 download (approx 4 GB):

    1. Download the zipped dataset from the authors' Google Drive page:
       https://github.com/cvlab-stonybrook/LearningToCountEverything
       (see "Dataset FSC-147" section).
    2. Unzip to: {target}
    3. Expected layout afterward:
       {target}/images_384_VarV2/*.jpg
       {target}/annotation_FSC147_384.json
       {target}/Train_Test_Val_FSC_147.json

    Alternatively, if `gdown` is installed:
       pip install gdown
       gdown --folder <google-drive-folder-id> -O {target}
    Replace <google-drive-folder-id> with the public folder ID from the
    authors' README.
""")

SAM_VIT_H_URL = (
    "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
)


def print_fsc147_instructions(target: str | Path) -> None:
    print(FSC147_INSTRUCTIONS.format(target=str(Path(target).resolve())))


def print_sam_vit_h_instructions(target: str | Path) -> None:
    p = Path(target).resolve()
    print(dedent(f"""\
        SAM ViT-H checkpoint (approx 2.4 GB):

            mkdir -p {p.parent}
            curl -L -o {p} {SAM_VIT_H_URL}

        Verify:
            ls -lh {p}
    """))

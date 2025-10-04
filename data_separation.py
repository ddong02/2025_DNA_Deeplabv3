
import os
import shutil
from pathlib import Path
from glob import glob

def split_dataset(base_dir):
    """
    이미지 기준으로 7:2:1로 train/val/test를 분할하되,
    labelmap/colormap은 파일명 패턴을 맞춰서 함께 이동한다.
    - image 파일:       000001_leftImg8bit.png
    - labelmap 파일 예: 000001_gtFine_CategoryId.png
    - colormap 파일 예: 000001_gtFine_Color.png (필요시 변경)
    """
    # ====== 폴더/파일명 패턴 설정 ======
    main_folders = ['colormap', 'image', 'labelmap']

    # 이미지 파일 접미사(기준 파일명). ex) *_leftImg8bit.png
    IMAGE_SUFFIX = "_leftImg8bit.png"

    # 라벨/컬러맵 접미사(이미지의 베이스이름 + 아래 접미사로 찾음)
    LABELMAP_SUFFIX = "_gtFine_CategoryId.png"   # 필요 시 수정
    COLORMAP_SUFFIX = "_gtFine_Color.png"        # 필요 시 수정

    # 매핑 함수: image 파일명 -> 각 폴더에서의 대응 파일명
    def map_filename_for_folder(main_folder: str, img_filename: str) -> str | None:
        """
        image 파일명(예: 000001_leftImg8bit.png)을
        폴더별 매핑된 파일명(예: labelmap -> 000001_gtFine_CategoryId.png)으로 변환.
        매핑 규칙을 모르면 None 반환(이후 glob fallback 사용).
        """
        if not img_filename.endswith(IMAGE_SUFFIX):
            return None
        base = img_filename[:-len(IMAGE_SUFFIX)]  # "000001"
        if main_folder == 'image':
            return img_filename
        elif main_folder == 'labelmap':
            return base + LABELMAP_SUFFIX
        elif main_folder == 'colormap':
            return base + COLORMAP_SUFFIX
        return None

    # ====== 분할 기준 경로 ======
    source_base_path = Path(base_dir) / 'image' / 'train'
    if not source_base_path.exists():
        print(f"오류: 기준 경로 '{source_base_path}'를 찾을 수 없습니다.")
        print("base_directory 변수가 올바르게 설정되었는지 확인하세요.")
        return

    try:
        sub_folders = [d.name for d in source_base_path.iterdir() if d.is_dir()]
    except OSError as e:
        print(f"오류: '{source_base_path}' 폴더를 읽는 중 문제가 발생했습니다: {e}")
        return

    print("데이터셋 분할을 시작합니다...")

    for sub_folder in sub_folders:
        print(f"\n📁 [{sub_folder}] 폴더 처리 중...")

        source_sub_folder_path = source_base_path / sub_folder
        try:
            files = sorted([f.name for f in source_sub_folder_path.iterdir() if f.is_file()])
        except FileNotFoundError:
            print(f"  '{source_sub_folder_path}' 폴더를 찾을 수 없어 건너뜁니다.")
            continue

        if not files:
            print(f"  '{sub_folder}' 폴더에 파일이 없어 건너뜁니다.")
            continue

        # 목적지 폴더 생성
        for main_folder in main_folders:
            for split_type in ['train', 'val', 'test']:
                dest_path = Path(base_dir) / main_folder / split_type / sub_folder
                dest_path.mkdir(parents=True, exist_ok=True)

        # 통계
        moved_counts = {'train': 0, 'val': 0, 'test': 0}
        missing_counts = {'colormap': 0, 'labelmap': 0}  # 매핑 실패/파일 없음 카운트

        # 10개 묶음 단위로 7/2/1 분할
        for i in range(0, len(files), 10):
            chunk = files[i:i+10]

            if len(chunk) < 10:
                split_map = {'train': chunk}
            else:
                split_map = {
                    'train': chunk[0:7],
                    'val': chunk[7:9],
                    'test': chunk[9:10]
                }

            for split_type, files_to_move in split_map.items():
                if not files_to_move:
                    continue

                for img_name in files_to_move:
                    moved_counts[split_type] += 1

                    # 각 폴더별로 대응 파일명 찾아서 이동
                    for main_folder in main_folders:
                        # 원본 파일이 위치한 "train" 폴더(초기 전체가 train에 있다 가정)
                        src_dir = Path(base_dir) / main_folder / 'train' / sub_folder

                        # 1) 규칙 매핑으로 시도
                        mapped = map_filename_for_folder(main_folder, img_name)
                        if mapped:
                            source_file = src_dir / mapped
                            if source_file.exists():
                                dest_file = Path(base_dir) / main_folder / split_type / sub_folder / mapped
                                shutil.move(str(source_file), str(dest_file))
                                continue  # 다음 폴더로

                        # 2) 매핑 실패 또는 해당 파일 없음 → glob로 유연하게 탐색
                        #    이미지의 base로 시작하는 파일을 한 개만 찾으면 그것을 이동
                        if img_name.endswith(IMAGE_SUFFIX):
                            base = img_name[:-len(IMAGE_SUFFIX)]
                        else:
                            base = os.path.splitext(img_name)[0]

                        # 대표적인 후보 패턴들 (필요시 추가)
                        patterns = [
                            f"{base}*.png",
                            f"{base}*.jpg",
                            f"{base}*.jpeg",
                            f"{base}*.bmp"
                        ]

                        found = None
                        for pat in patterns:
                            matches = glob(str(src_dir / pat))
                            # 후보가 딱 1개면 그것을 채택
                            if len(matches) == 1:
                                found = matches[0]
                                break
                            # 여러 개면 규칙 우선순위로 거르기
                            if len(matches) > 1:
                                # labelmap/colormap에 대해 자주 쓰는 접미사 우선
                                prefer = []
                                if main_folder == 'labelmap':
                                    prefer = [LABELMAP_SUFFIX]
                                elif main_folder == 'colormap':
                                    prefer = [COLORMAP_SUFFIX]
                                else:
                                    prefer = [IMAGE_SUFFIX]

                                chosen = None
                                for m in matches:
                                    name = os.path.basename(m)
                                    if any(name.endswith(suf) for suf in prefer):
                                        chosen = m
                                        break
                                found = chosen or matches[0]
                                break

                        if found:
                            source_file = Path(found)
                            dest_name = os.path.basename(found)
                            dest_file = Path(base_dir) / main_folder / split_type / sub_folder / dest_name
                            shutil.move(str(source_file), str(dest_file))
                        else:
                            # 못 찾았다고 기록 (image 폴더는 반드시 있어야 하므로 여기로 오기 어려움)
                            if main_folder in missing_counts:
                                missing_counts[main_folder] += 1

        print(f"  - ✅ Train: {moved_counts['train']}개 파일 이동 완료")
        print(f"  - ✅ Validation: {moved_counts['val']}개 파일 이동 완료")
        print(f"  - ✅ Test: {moved_counts['test']}개 파일 이동 완료")
        if sum(missing_counts.values()) > 0:
            print(f"  - ⚠ 일부 매칭 실패: {missing_counts}")

    print("\n🎉 모든 작업이 완료되었습니다.")


# --- 실행 예시 ---
if __name__ == '__main__':
    base_directory = 'datasets/data/SemanticDataset_final'
    split_dataset(base_directory)

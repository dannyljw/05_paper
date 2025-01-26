#!/bin/bash

# 원격 리포지토리 URL
REPO_URL="https://github.com/dannyljw/05_paper.git"

# Git 리포지토리 초기화 및 원격 설정
if [ ! -d .git ]; then
    echo "Initializing Git repository..."
    git init
    git remote add origin "$REPO_URL"
fi

# 브랜치 설정
if ! git show-ref --verify --quiet refs/heads/main; then
    echo "Creating main branch..."
    git branch -M main
fi

# 모든 .log 파일을 찾아 반복 처리
find . -type f -name "*.log" | while read -r log_file; do
    # ZIP 파일 이름 생성 (기존 파일명에 .zip 확장자 추가)
    zip_file="${log_file%.log}.zip"

    # .log 파일을 ZIP 파일로 압축
    echo "Compressing $log_file to $zip_file..."
    zip -j "$zip_file" "$log_file"

    # 원본 .log 파일 삭제
    if [ -f "$zip_file" ]; then
        echo "Deleting original file $log_file..."
        rm "$log_file"
    else
        echo "Failed to compress $log_file. Skipping deletion."
    fi
done

# Git에 변경 사항 추가
echo "Adding changes to Git..."
git add .

# 커밋 메시지
echo "Committing changes..."
git commit -m "Compressed .log files into individual .zip files and removed originals."

# 변경 사항 푸시
echo "Pushing to GitHub..."
git push -u origin main

echo "Process completed."

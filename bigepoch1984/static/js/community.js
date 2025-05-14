function closeUpdateBox() {
  const container = document.getElementById('update-container');
  if (container) container.style.display = 'none';
}

function openFilterPopup() {
  const popup = document.getElementById('filter-popup');
  popup.style.display = 'block';
  setTimeout(() => popup.classList.add('show'), 10);
}

function closeFilterPopup() {
  const popup = document.getElementById('filter-popup');
  popup.classList.remove('show');
  setTimeout(() => popup.style.display = 'none', 300);
}

function selectPeriod(period) {
  document.querySelectorAll('#period-options .filter-option').forEach(option => {
    option.classList.toggle('active', option.textContent === period);
  });
}

function selectSort(sort) {
  document.querySelectorAll('#sort-options .filter-option').forEach(option => {
    option.classList.toggle('active', option.textContent === sort);
  });
}

function confirmFilter() {
  const period = document.querySelector('#period-options .filter-option.active')?.textContent || '한달';
  const sort = document.querySelector('#sort-options .filter-option.active')?.textContent || '최신순';
  const url = new URL(window.location);
  url.searchParams.set('period', period);
  url.searchParams.set('sort', sort);
  window.location.href = url.toString();
}

function openSharePopup() {
  document.getElementById('share-popup').style.display = 'block';
}

function closeSharePopup() {
  document.getElementById('share-popup').style.display = 'none';
}

function goShare(platform) {
  const url = window.location.href;
  let shareUrl = '';
  switch (platform) {
    case 'kakao':
      shareUrl = `https://story.kakao.com/s/share?url=${encodeURIComponent(url)}`;
      break;
    case 'telegram':
      shareUrl = `https://t.me/share/url?url=${encodeURIComponent(url)}`;
      break;
    case 'facebook':
      shareUrl = `https://www.facebook.com/sharer/sharer.php?u=${encodeURIComponent(url)}`;
      break;
    case 'twitter':
      shareUrl = `https://twitter.com/intent/tweet?url=${encodeURIComponent(url)}`;
      break;
  }
  window.open(shareUrl, '_blank');
}

document.addEventListener('DOMContentLoaded', () => {
  const csrfToken = document.querySelector('[name=csrfmiddlewaretoken]').value;
  const likeBtn = document.querySelector('.like-btn');
  const worryBtn = document.querySelector('.worry-btn');
  const blockModal = new bootstrap.Modal(document.getElementById('blockModal'));
  const blockMessage = document.getElementById('blockMessage');
  const confirmBlock = document.getElementById('confirmBlock');
  
  const body = document.querySelector('body');
  const blockUserUrl = body.dataset.blockUrl;
  const reportUserUrl = body.dataset.reportUrl; // 신고 URL, 없으면 사용 안 함
  const likePostUrl = body.dataset.likeUrl;
  const commentCreateUrl = body.dataset.commentCreateUrl;
  const commentDeleteUrlTemplate = body.dataset.commentDeleteUrl;
  const communityUrl = body.dataset.communityUrl;

  // 좋아요/걱정돼요 AJAX
  document.querySelectorAll('.like-btn, .worry-btn').forEach(button => {
    button.addEventListener('click', function(e) {
      e.preventDefault();
      const postId = this.getAttribute('data-post-id');
      const action = this.getAttribute('data-action');

      console.log(`Sending request: postId=${postId}, action=${action}`);

      fetch(likePostUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
          'X-CSRFToken': csrfToken
        },
        body: `action=${encodeURIComponent(action)}`
      })
      .then(response => {
        console.log('Response status:', response.status);
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
      })
      .then(data => {
        console.log('Response data:', data);
        if (data.status === 'success') {
          if (likeBtn) {
            likeBtn.querySelector('span').textContent = data.likes_count;
            likeBtn.classList.toggle('liked', data.is_liked);
          }
          if (worryBtn) {
            worryBtn.querySelector('span').textContent = data.worried_count;
            worryBtn.classList.toggle('worried', data.is_worried);
          }
        } else {
          console.error('Server error:', data.error);
          alert(data.error || '요청 처리 중 오류가 발생했습니다.');
        }
      })
      .catch(error => {
        console.error('Fetch error:', error);
        alert('서버와의 연결에 문제가 발생했습니다.');
      });
    });
  });

  // 댓글 작성 AJAX
  document.getElementById('comment-form').addEventListener('submit', function(e) {
    e.preventDefault();
    const postId = this.getAttribute('data-post-id');
    const content = this.querySelector('input[name="content"]').value;
    fetch(commentCreateUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-CSRFToken': csrfToken
      },
      body: JSON.stringify({ content: content })
    })
    .then(response => {
      console.log('Comment response status:', response.status);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      return response.json();
    })
    .then(data => {
      console.log('Comment response data:', data);
      if (data.status === 'success') {
        const commentList = document.getElementById('comment-list');
        const commentDiv = document.createElement('div');
        commentDiv.className = 'comment d-flex justify-content-between align-items-start';
        commentDiv.setAttribute('data-comment-id', data.comment.id);
        commentDiv.innerHTML = `
          <div>
            <div class="profile-preview me-2">
              <i class="bi bi-person-fill profile-icon"></i>
            </div>
            <strong>${data.comment.user.nickname}</strong>
            <span class="badge bg-primary ms-1">${data.comment.user.auth_id === 'admin' ? '운영자' : '일반회원'}</span><br>
            <span>${data.comment.content}</span><br>
            <small class="text-muted">방금 전</small>
          </div>
          <div class="d-flex gap-2">
            <a href="/community/comment/edit/${data.comment.id}/" class="btn btn-outline-secondary btn-sm">
              <i class="bi bi-pencil"></i>
            </a>
            <button class="btn btn-outline-secondary btn-sm delete-comment-btn" data-comment-id="${data.comment.id}">
              <i class="bi bi-trash"></i>
            </button>
          </div>
        `;
        if (commentList.querySelector('.text-muted')) {
          commentList.querySelector('.text-muted').remove();
        }
        commentList.insertBefore(commentDiv, commentList.firstChild);
        document.querySelector('.comment-box strong').textContent = `${data.comments_count}개의 댓글`;
        this.querySelector('input[name="content"]').value = '';
      } else {
        console.error('Comment error:', data.error);
        alert(data.error || '댓글 작성 중 오류가 발생했습니다.');
      }
    })
    .catch(error => {
      console.error('Comment fetch error:', error);
      alert('서버와의 연결에 문제가 발생했습니다.');
    });
  });

  // 댓글 삭제 AJAX
  document.querySelectorAll('.delete-comment-btn').forEach(button => {
    button.addEventListener('click', function(e) {
      e.preventDefault();
      const commentId = this.getAttribute('data-comment-id');
      if (confirm('이 댓글을 삭제하시겠습니까?')) {
        const commentDeleteUrl = commentDeleteUrlTemplate.replace('0', commentId);
        fetch(commentDeleteUrl, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': csrfToken
          }
        })
        .then(response => {
          console.log('Delete comment response status:', response.status);
          if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
          }
          return response.json();
        })
        .then(data => {
          console.log('Delete comment response data:', data);
          if (data.status === 'success') {
            document.querySelector(`.comment[data-comment-id="${commentId}"]`).remove();
            document.querySelector('.comment-box strong').textContent = `${data.comments_count}개의 댓글`;
            if (document.getElementById('comment-list').children.length === 0) {
              const noComments = document.createElement('div');
              noComments.className = 'text-muted';
              noComments.textContent = '아직 댓글이 없습니다.';
              document.getElementById('comment-list').appendChild(noComments);
            }
          } else {
            console.error('Delete comment error:', data.error);
            alert(data.error || '댓글 삭제 중 오류가 발생했습니다.');
          }
        })
        .catch(error => {
          console.error('Delete comment fetch error:', error);
          alert('서버와의 연결에 문제가 발생했습니다.');
        });
      }
    });
  });

  // 신고/차단 버튼 이벤트
  document.querySelectorAll('.dropdown-item[data-action]').forEach(button => {
    button.addEventListener('click', function(e) {
      e.preventDefault();
      const action = this.getAttribute('data-action');
      const targetUser = this.getAttribute('data-user');
      const postId = this.getAttribute('data-post-id');

      if (!confirm(`${targetUser}님을 ${action === 'report' ? '신고' : '차단'}하시겠습니까?`)) {
        return;
      }

      let url = '';
      if (action === 'report') {
        url = reportUserUrl; // 신고 뷰가 없으면 사용 안 함
      } else if (action === 'block') {
        url = blockUserUrl;
      }

      console.log(`Fetching URL: ${url}`);

      fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
          'X-CSRFToken': csrfToken
        },
        body: `target_user=${encodeURIComponent(targetUser)}`
      })
      .then(response => {
        console.log('Response status:', response.status);
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
      })
      .then(data => {
        console.log('Response data:', data);
        if (data.status === 'success') {
          if (action === 'block') {
            blockMessage.textContent = `${targetUser}님을 차단했습니다.`;
            blockModal.show();
            confirmBlock.onclick = () => {
              window.location.href = communityUrl;
            };
          } else {
            alert(`${targetUser}님을 ${action === 'report' ? '신고' : '차단'}했습니다.`);
          }
        } else {
          alert(data.error || '처리 중 오류가 발생했습니다.');
        }
      })
      .catch(error => {
        console.error('Error:', error);
        alert('서버와의 연결에 문제가 발생했습니다.');
      });
    });
  });
});
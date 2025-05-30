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
  const csrfToken = document.querySelector('[name=csrfmiddlewaretoken]')?.value;
  if (!csrfToken) {
    console.error('CSRF token not found');
    return;
  }

  const body = document.querySelector('body');
  const likePostUrl = body.dataset.likeUrl;
  const commentDeleteUrlTemplate = body.dataset.commentDeleteUrl;

  // 좋아요/걱정 AJAX
  document.querySelectorAll('.like-btn, .worry-btn').forEach(button => {
    button.addEventListener('click', function(e) {
      e.preventDefault();
      if (body.dataset.isAuthenticated !== 'true') {
        window.location.href = body.dataset.loginUrl;
        return;
      }

      const postId = this.getAttribute('data-post-id');
      const action = this.getAttribute('data-action');

      if (!likePostUrl) {
        console.error('Like post URL not defined');
        return;
      }

      fetch(likePostUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
          'X-CSRFToken': csrfToken
        },
        body: `action=${encodeURIComponent(action)}`
      })
      .then(response => {
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
      })
      .then(data => {
        if (data.status === 'success') {
          const likeBtn = document.querySelector(`.like-btn[data-post-id="${postId}"]`);
          const worryBtn = document.querySelector(`.worry-btn[data-post-id="${postId}"]`);
          if (likeBtn) {
            likeBtn.querySelector('span').textContent = data.likes_count;
            likeBtn.classList.toggle('liked', data.is_liked);
          }
          if (worryBtn) {
            worryBtn.querySelector('span').textContent = data.worried_count;
            worryBtn.classList.toggle('worried', data.is_worried);
          }
        } else {
          console.error('Like/Worry error:', data.error);
          alert(data.error || '요청 처리 중 오류가 발생했습니다.');
        }
      })
      .catch(error => {
        console.error('Fetch error:', error);
        alert('서버와의 연결에 문제가 발생했습니다. 다시 시도해주세요.');
      });
    });
  });

  // 댓글 삭제 AJAX
  document.querySelectorAll('.delete-comment-btn').forEach(button => {
    button.addEventListener('click', function(e) {
      e.preventDefault();
      if (!confirm('댓글을 삭제하시겠습니까?')) return;

      const commentId = this.getAttribute('data-comment-id');
      const url = commentDeleteUrlTemplate.replace('0', commentId);

      fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
          'X-CSRFToken': csrfToken
        }
      })
      .then(response => {
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
      })
      .then(data => {
        if (data.status === 'success') {
          const commentElement = document.querySelector(`.comment[data-comment-id="${data.comment_id}"]`);
          if (commentElement) commentElement.remove();
          document.querySelector('.comment-box strong').textContent = `${data.comments_count}개의 댓글`;
        } else {
          console.error('Comment delete error:', data.error);
          alert(data.error || '댓글 삭제에 실패했습니다.');
        }
      })
      .catch(error => {
        console.error('Fetch error:', error);
        alert('서버와의 연결에 문제가 발생했습니다. 다시 시도해주세요.');
      });
    });
  });
});
document.addEventListener('DOMContentLoaded', () => {
  const scrollTopBtn = document.getElementById('scrollTopBtn');
  window.addEventListener('scroll', () => {
    if (window.scrollY > 200) {
      scrollTopBtn.style.display = 'block';
    } else {
      scrollTopBtn.style.display = 'none';
    }
  });
  scrollTopBtn.addEventListener('click', () => {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  });
});

function toggleLike(button) {
  console.log('toggleLike called for post:', button.getAttribute('data-post-id'));
  const postId = button.getAttribute('data-post-id');
  const errorDiv = document.getElementById('like-error');
  button.disabled = true;
  errorDiv.style.display = 'none';

  const csrfToken = getCsrfToken();
  if (!csrfToken) {
    console.error('CSRF token not found');
    errorDiv.textContent = 'CSRF 토큰을 찾을 수 없습니다.';
    errorDiv.style.display = 'block';
    button.disabled = false;
    return;
  }

  console.log('Sending AJAX request to /community/' + postId + '/like/');
  fetch(`/community/${postId}/like/`, {
    method: 'POST',
    headers: {
      'X-CSRFToken': csrfToken,
      'Content-Type': 'application/json',
    },
  })
    .then(response => {
      console.log('Response status:', response.status);
      return response.json();
    })
    .then(data => {
      console.log('Response data:', data);
      if (data.error) {
        errorDiv.textContent = data.error;
        errorDiv.style.display = 'block';
      } else {
        button.querySelector('span').textContent = data.likes_count;
        if (data.is_liked) {
          button.classList.add('liked');
        } else {
          button.classList.remove('liked');
        }
      }
      button.disabled = false;
    })
    .catch(error => {
      console.error('AJAX error:', error);
      errorDiv.textContent = '좋아요 처리 중 오류가 발생했습니다.';
      errorDiv.style.display = 'block';
      button.disabled = false;
    });
}

function getCsrfToken() {
  let token = document.querySelector('meta[name="csrf-token"]')?.content;
  if (!token) {
    token = document.querySelector('input[name="csrfmiddlewaretoken"]')?.value;
  }
  console.log('CSRF token:', token);
  return token;
}
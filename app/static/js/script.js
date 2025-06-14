const previewImage = document.getElementById("previewImage");

imageInput.addEventListener("change", (event) => {
  const file = event.target.files[0];
  if (file) {
    const imageURL = URL.createObjectURL(file);
    previewImage.src = imageURL;
  }
});

async function sendData() {
  const imageInput = document.getElementById("imageInput");
  const questionInput = document.getElementById("questionInput");

  const imageFile = imageInput.files[0];
  const questionText = questionInput.value;

  if (!imageFile || !questionText) {
    alert("Please select an image and enter a question")
    return;
  }

  const formData = new FormData();
  formData.append("image", imageFile);
  formData.append("question", questionText);

  try {
    const response = await fetch("/api/send", {
      method: "POST",
      body: formData,
    })

    const data = await response.json();
    receiveData(data);
    console.log("Phản hồi từ Flask:", data);
  } catch {
    console.error("Lỗi:", error);
  }
}

function receiveData(data) {
  const resultContent = document.getElementById("resultContent");

  resultContent.innerHTML = `
  - <b>Câu hỏi:</b> ${data.question}<br >
  - <b>Tên ảnh:</b> ${data.image_filename} <br>
  - <b>Đáp án:</b> ${data.predicted_answer} <br>
`;
}
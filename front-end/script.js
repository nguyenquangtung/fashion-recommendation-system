// Dữ liệu danh sách sản phẩm (số lượng sản phẩm và tên ảnh)
const productData = {
  totalProducts: 100,
  productsPerPage: 10,
  currentPage: 1,
};

const imageOrder = [3, 1, 5, 2, 4]; // Định danh thứ tự của các hình ảnh
const totalProducts = imageOrder.length; // Số lượng hình ảnh

function displayProducts() {
  const productList = document.getElementById("product-list");
  productList.innerHTML = "";

  for (let i = 0; i < totalProducts; i++) {
    const productDiv = document.createElement("div");
    productDiv.classList.add("product");

    // Tạo một thẻ <img> cho sản phẩm
    const productImage = document.createElement("img");
    productImage.src = `testdata/product${imageOrder[i]}.jpg`; // Sử dụng danh sách định danh
    productDiv.appendChild(productImage);

    // Tạo tên sản phẩm (điều này cần được điều chỉnh cho dữ liệu thực tế)
    const productName = document.createElement("h2");
    productName.textContent = `Sản phẩm ${imageOrder[i]}`;
    productDiv.appendChild(productName);

    productList.appendChild(productDiv);
  }

  displayPagination();
}

// Các hàm khác không thay đổi
function displayProducts() {
  const productList = document.getElementById("product-list");
  productList.innerHTML = "";

  const startIndex =
    (productData.currentPage - 1) * productData.productsPerPage;
  const endIndex = startIndex + productData.productsPerPage;

  for (let i = startIndex; i < endIndex && i < productData.totalProducts; i++) {
    const productDiv = document.createElement("div");
    productDiv.classList.add("product");

    // Tạo một thẻ <img> cho sản phẩm (điều này cần được điều chỉnh cho dữ liệu thực tế)
    const productImage = document.createElement("img");
    productImage.src = `fashion-recommendation-system/testdata/${i + 1}.jpg`; // Đường dẫn đến thư mục testdata
    productDiv.appendChild(productImage);

    // Tạo tên sản phẩm (điều này cần được điều chỉnh cho dữ liệu thực tế)
    const productName = document.createElement("h2");
    productName.textContent = `Sản phẩm ${i + 1}`;
    productDiv.appendChild(productName);

    productList.appendChild(productDiv);
  }

  displayPagination();
}

// Hàm hiển thị phân trang
function displayPagination() {
  const pagination = document.getElementById("pagination");
  pagination.innerHTML = "";

  const totalPages = Math.ceil(
    productData.totalProducts / productData.productsPerPage
  );

  for (let i = 1; i <= totalPages; i++) {
    const pageLink = document.createElement("a");
    pageLink.href = "#";
    pageLink.textContent = i;
    pageLink.addEventListener("click", () => {
      productData.currentPage = i;
      displayProducts();
    });

    pagination.appendChild(pageLink);
  }
}

// Hiển thị sản phẩm khi trang web được tải
displayProducts();

import axios from "axios";

// Read API URL from environment variables, defaulting to local dev endpoint
const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api/v1";

const apiClient = axios.create({
  baseURL: API_URL,
  headers: {
    "Content-Type": "application/json",
  },
  timeout: 10000, // 10s request timeout limit
});

apiClient.interceptors.request.use(
  (config) => {
    if (typeof window !== "undefined") {
      const token = localStorage.getItem("access_token");
      if (token && config.headers) {
        config.headers.Authorization = `Bearer ${token}`;
      }
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Gracefully handle token expiration or unauthorized exceptions globally
apiClient.interceptors.response.use(
  (response) => response,
  async (error) => {
    const originalRequest = error.config;
    
    // If error is 401 and we haven't already retried this request
    if (error.response && error.response.status === 401 && !originalRequest._retry) {
      originalRequest._retry = true;
      
      if (typeof window !== "undefined") {
        const refreshToken = localStorage.getItem("refresh_token");
        
        if (refreshToken) {
          try {
            // Attempt to refresh the token
            const response = await axios.post(`${API_URL}/auth/refresh`, {
              refresh_token: refreshToken
            });
            
            if (response.data && response.data.access_token) {
              // Store new tokens
              localStorage.setItem("access_token", response.data.access_token);
              
              // Retry the original request
              originalRequest.headers.Authorization = `Bearer ${response.data.access_token}`;
              return apiClient(originalRequest);
            }
          } catch (refreshError) {
            // Refresh failed, clear tokens and redirect to login
            localStorage.removeItem("access_token");
            localStorage.removeItem("refresh_token");
            if (!window.location.pathname.endsWith("/login")) {
              window.location.href = "/login";
            }
          }
        } else {
          // No refresh token available, redirect to login
          localStorage.removeItem("access_token");
          localStorage.removeItem("refresh_token");
          if (!window.location.pathname.endsWith("/login")) {
            window.location.href = "/login";
          }
        }
      }
    }
    return Promise.reject(error);
  }
);

export default apiClient;

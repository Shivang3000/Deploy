import axios from 'axios'

const baseURL = import.meta.env.VITE_BACKEND_BASE_API

let onLogout = null;

export const setLogoutHandler = (callback) => {
    onLogout = callback;
};

const axiosInstance = axios.create({
    baseURL,
    headers: {
        "Content-Type": "application/json"
    }
});

// Request interceptor
axiosInstance.interceptors.request.use(config => {
    const accessToken = localStorage.getItem('accessToken');
    if (accessToken) {
        config.headers.Authorization = `Bearer ${accessToken}`;
    }
    return config;
});

// Response interceptor
axiosInstance.interceptors.response.use(
    response => response,
    async error => {
        const originalRequest = error.config;

        if (
            error.response?.status === 401 &&
            !originalRequest._retry &&
            !originalRequest.url.includes('/token/refresh/')
        ) {
            originalRequest._retry = true;

            try {
                const refreshToken = localStorage.getItem('refreshToken');
                const res = await axiosInstance.post('/token/refresh/', {
                    refresh: refreshToken
                });

                localStorage.setItem('accessToken', res.data.access);
                originalRequest.headers.Authorization = `Bearer ${res.data.access}`;
                return axiosInstance(originalRequest);

            } catch (err) {
                localStorage.removeItem('accessToken');
                localStorage.removeItem('refreshToken');

                if (onLogout) onLogout(); // notify React
            }
        }

        return Promise.reject(error);
    }
);

export default axiosInstance;

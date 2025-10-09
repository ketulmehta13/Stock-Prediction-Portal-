import axios from "axios";

// Use environment variable with proper fallback
const baseURL = import.meta.env.VITE_BACKEND_BASE_API || 'http://127.0.0.1:8000/api/v1/';

const axiosInstance = axios.create({
    baseURL: baseURL,
    headers: {
        'Content-Type': 'application/json',
    }
})

// Request Interceptor
axiosInstance.interceptors.request.use(
    function(config){
        const accessToken = localStorage.getItem('accessToken')
        if(accessToken){
            config.headers['Authorization'] = `Bearer ${accessToken}`
        }
        return config;
    },
    function(error){
        return Promise.reject(error);
    }
)

// Response Interceptor
axiosInstance.interceptors.response.use(
    function(response){
        return response;
    },
    // Handle failed responses
    async function(error){
        const originalRequest = error.config;
        if(error.response && error.response.status === 401 && !originalRequest.retry){
            originalRequest.retry = true;
            const refreshToken = localStorage.getItem('refreshToken')
            try{
                const response = await axiosInstance.post('/token/refresh/', {refresh: refreshToken})
                localStorage.setItem('accessToken', response.data.access)
                originalRequest.headers['Authorization'] = `Bearer ${response.data.access}`
                return axiosInstance(originalRequest)
            }catch(error){
                localStorage.removeItem('accessToken')
                localStorage.removeItem('refreshToken')
                // Optional: redirect to login page
                // window.location.href = '/login';
            }
        }
        return Promise.reject(error);
    }
)

export default axiosInstance;

import { createContext, useState, useEffect } from 'react'
import { setLogoutHandler } from './axiosInstance'

const AuthContext = createContext();

const AuthProvider = ({ children }) => {

    const [isLoggedIn, setIsLoggedIn] = useState(
        !!localStorage.getItem('accessToken')
    );

    useEffect(() => {
        setLogoutHandler(() => {
            setIsLoggedIn(false);
        });
    }, []);

    return (
        <AuthContext.Provider value={{ isLoggedIn, setIsLoggedIn }}>
            {children}
        </AuthContext.Provider>
    );
};

export default AuthProvider;
export { AuthContext };

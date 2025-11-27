import React from 'react';
import { useI18n } from '../i18n/I18nContext';
import LanguageSwitcher from './LanguageSwitcher';

interface LayoutProps {
    children: React.ReactNode;
}

const Layout: React.FC<LayoutProps> = ({ children }) => {
    const { t } = useI18n();

    return (
        <div className="min-h-screen flex flex-col bg-slate-950 text-slate-100">
            <header className="border-b border-slate-800 bg-slate-900/70 backdrop-blur">
                <div className="max-w-4xl mx-auto px-4 py-4">
                    <div className="flex items-center justify-between">
                        <div>
                            <h1 className="text-3xl font-bold">
                                {t("appTitle")}
                            </h1>
                            <p className="text-sm text-slate-400 mt-1">
                                {t("appSubtitle")}
                            </p>
                        </div>
                        <LanguageSwitcher />
                    </div>
                </div>
            </header>
            
            <main className="flex-1 flex justify-center px-4 py-8">
                <div className="w-full max-w-4xl space-y-6">
                    {children}
                </div>
            </main>
            
            <footer className="border-t border-slate-800 text-xs text-slate-400 py-3 text-center">
                {t("footerNote")}
            </footer>
        </div>
    );
};

export default Layout;

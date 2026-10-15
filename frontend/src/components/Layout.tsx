/** App shell: header, navigation, and the educational-use disclaimer. */

import { NavLink, Outlet } from 'react-router-dom';

const NAV_ITEMS = [
  { to: '/', label: 'Dashboard' },
  { to: '/predict', label: 'Predict' },
  { to: '/result', label: 'Result' },
  { to: '/model', label: 'Model info' },
] as const;

export function Layout() {
  return (
    <div className="flex min-h-screen flex-col bg-slate-50 text-slate-900">
      <header className="border-b border-slate-200 bg-white">
        <div className="mx-auto w-full max-w-5xl px-4 py-4 sm:px-6">
          <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
            <div>
              <p className="text-sm font-semibold tracking-tight text-slate-900">
                Loan Delinquency Risk
              </p>
              <p className="text-xs text-slate-500">
                Random Forest · Give Me Some Credit · educational prototype
              </p>
            </div>
            <nav aria-label="Main" className="-mx-1 flex flex-wrap gap-y-1">
              {NAV_ITEMS.map((item) => (
                <NavLink
                  key={item.to}
                  to={item.to}
                  end={item.to === '/'}
                  className={({ isActive }) =>
                    `mx-0.5 rounded-md px-2.5 py-1.5 text-sm font-medium transition sm:mx-1 sm:px-3 ${
                      isActive
                        ? 'bg-indigo-50 text-indigo-700'
                        : 'text-slate-600 hover:bg-slate-100 hover:text-slate-900'
                    }`
                  }
                >
                  {item.label}
                </NavLink>
              ))}
            </nav>
          </div>
        </div>
      </header>

      <main className="mx-auto w-full max-w-5xl flex-1 px-4 py-8 sm:px-6">
        <Outlet />
      </main>

      <footer className="border-t border-slate-200 bg-white">
        <div className="mx-auto w-full max-w-5xl px-4 py-5 text-xs leading-relaxed text-slate-500 sm:px-6">
          Educational student project. Predictions are model estimates on a
          2011 public dataset and must not be used to approve, decline or price
          real credit.
        </div>
      </footer>
    </div>
  );
}
